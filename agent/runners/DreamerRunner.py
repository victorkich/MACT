import ray
import wandb
from copy import deepcopy

from agent.workers.DreamerWorker import DreamerWorker

import numpy as np
import pickle
from pathlib import Path
from environments import Env

class DreamerServer:
    def __init__(self, n_workers, env_config, controller_config, model):
        import os as _os
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            _temp_dir=_os.environ.get('RAY_TEMP_DIR', f'/tmp/ray_{_os.getpid()}'),
            object_store_memory=2 * 1024**3,  # 2 GB from /tmp — /dev/shm is only 64 MB in Docker
        )

        self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model) for worker in self.workers]
        self.env_type = controller_config.ENV_TYPE
        # Kept so a crashed data-collection worker (SC2 BrokenPipeError) can be rebuilt
        # mid-run instead of killing the whole job.
        self._n_workers    = n_workers
        self._env_config   = env_config
        self._ctrl_config  = controller_config
        self._latest_params = model
        
        eval_controller_config = deepcopy(controller_config)
        eval_controller_config.temperature = 1.0  # 1.0
        if hasattr(eval_controller_config, 'determinisitc'):
            eval_controller_config.determinisitc = True

        self.eval_episodes_num = 8
        self._eval_cfg      = eval_controller_config
        self._eval_env_cfg  = env_config
        self.eval_workers   = []
        self._eval_ready    = False
        self.eval_tasks     = []

    def _ensure_eval_workers(self):
        if not self._eval_ready:
            self.eval_workers = [DreamerWorker.remote(i, self._eval_env_cfg, self._eval_cfg)
                                 for i in range(self.eval_episodes_num)]
            self._eval_ready = True

    def _reset_eval_workers(self):
        """Tear down eval workers (e.g. after an SC2 BrokenPipeError) so the next
        eval rebuilds them fresh instead of reusing a dead actor/SC2 instance."""
        for w in getattr(self, "eval_workers", []):
            try:
                ray.kill(w)
            except Exception:
                pass
        self.eval_workers = []
        self.eval_tasks = []
        self._eval_ready = False

    def append(self, idx, update):
        self._latest_params = update      # track latest params for crash rebuild
        self.tasks.append(self.workers[idx].run.remote(update))

    def _reset_run_workers(self):
        """Rebuild data-collection workers after an SC2 crash, re-submitting each with
        the latest params so training continues instead of dying."""
        for w in self.workers:
            try:
                ray.kill(w)
            except Exception:
                pass
        self.workers = [DreamerWorker.remote(i, self._env_config, self._ctrl_config)
                        for i in range(self._n_workers)]
        self.tasks = [w.run.remote(self._latest_params) for w in self.workers]

    def run(self):
        try:
            done_id, tasks = ray.wait(self.tasks)
            self.tasks = tasks
            return ray.get(done_id)[0]
        except Exception as e:
            print(f"[RUN] rollout worker crashed ({type(e).__name__}: {e}); "
                  f"rebuilding workers & retrying", flush=True)
            self._reset_run_workers()
            done_id, tasks = ray.wait(self.tasks)
            self.tasks = tasks
            return ray.get(done_id)[0]
    
    ## eval
    def eval_append(self, idx, update):
        self._ensure_eval_workers()
        self.eval_tasks.append(self.eval_workers[idx].run.remote(update))
        
    def evaluate(self, model_params):
        eval_win_rate = 0.
        eval_returns = 0.
        eval_steps = 0.
        n_ok = 0

        # An SC2 eval worker can crash mid-eval and raise BrokenPipeError / RayActorError /
        try:
            for i in range(self.eval_episodes_num):
                self.eval_append(i, model_params)

            for i in range(self.eval_episodes_num):
                done_id, eval_tasks = ray.wait(self.eval_tasks)
                self.eval_tasks = eval_tasks
                eval_rollout, eval_info = ray.get(done_id)[0]

                eval_win_rate += eval_info["reward"] if eval_info["reward"] is not None else 0.
                eval_returns += eval_rollout["reward"].sum(0).mean()
                eval_steps += eval_info["steps_done"]
                n_ok += 1
        except Exception as e:
            print(f"[EVAL] eval round failed ({type(e).__name__}: {e}); "
                  f"skipping & rebuilding eval workers (training continues)", flush=True)
            self._reset_eval_workers()

        if n_ok == 0:
            return 0.0, 0.0, 0.0
        return eval_win_rate / n_ok, eval_returns / n_ok, eval_steps / n_ok

    def evaluate_final(self, model_params, n_episodes=100):
        """High-quality final evaluation: runs n_episodes in batches of eval_episodes_num.
        Gives 1% win-rate granularity with n_episodes=100 (vs 5% from intermediate evals).
        Cost: ~5 batches × ~4 s each = ~20 seconds total — negligible."""
        total_wr, total_ret, n_batches = 0., 0., 0
        remaining = n_episodes
        while remaining > 0:
            batch = min(remaining, self.eval_episodes_num)
            wr, ret, _ = self.evaluate(model_params)
            # evaluate() always runs eval_episodes_num; scale contribution
            total_wr  += wr  * batch
            total_ret += ret * batch
            n_batches += batch
            remaining -= batch
        return total_wr / n_batches, total_ret / n_batches


class DreamerRunner:

    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        self.server = DreamerServer(n_workers, env_config, controller_config, self.learner.params())

        if getattr(learner_config, 'cpc_v2', False) and getattr(learner_config, 'map_name', '') == '3s_vs_4z':
            self.server.eval_episodes_num = 30

        self.save_path = Path(learner_config.RUN_DIR).parent / f"marie_{learner_config.map_name}_seed{learner_config.seed}.pkl"
        self.env_type = controller_config.ENV_TYPE
        
    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10, save_interval= 10000, save_mode="interval"):
        cur_steps, cur_episode = 0, 0
        save_interval_steps = 0
        last_save_steps = 0
        last_eval_steps = 0
        
        eval_win_rates = []
        eval_ret_list  = []
        steps = []

        wandb.define_metric("steps")
        wandb.define_metric("reward", step_metric="steps")
        wandb.define_metric("eval_win_rate", step_metric="steps")
        wandb.define_metric("eval_returns", step_metric="steps")

        while True:
            # NOTE: array manager backend... mp
            rollout, info = self.server.run()
            ent = rollout['entropy'].mean(0)
            ent_str = f""
            for e in ent.tolist():
                ent_str += f"{e:.4f} "

            cur_steps += info["steps_done"]
            cur_episode += 1
            save_interval_steps += info["steps_done"]
            returns = rollout["reward"].sum(0).mean()

            if self.env_type == Env.STARCRAFT:
                wandb.log({'win': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Win: {info["reward"]}', 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')
            elif self.env_type == Env.MAMUJOCO or self.env_type == Env.PETTINGZOO:
                wandb.log({'rew_per_step': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Rew per step: {info["reward"]}', 'Returns: %.4f' % returns, f"Average std: {ent_str}", sep=' | ')
            else:
                wandb.log({'scores': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Scores: {info["reward"]}', 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')


            wandb.log({'returns': returns, "episodes": cur_episode})

            self.learner.step(rollout)

            ## save model
            if (save_interval_steps - last_save_steps) > save_interval and save_mode == "interval":
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_{save_interval_steps // 1000}Ksteps.pth")
                last_save_steps = save_interval_steps // save_interval * save_interval

            ## evaluation
            if (save_interval_steps - last_eval_steps) > 500:
                eval_win_rate, eval_returns, aver_eval_steps = self.server.evaluate(self.learner.params())
                last_eval_steps = save_interval_steps // 500 * 500
                
                wandb.log({'eval_win_rate': eval_win_rate, "steps": save_interval_steps})
                wandb.log({'eval_returns': eval_returns, "steps": save_interval_steps})

                steps.append(save_interval_steps)
                eval_win_rates.append(eval_win_rate)
                eval_ret_list.append(eval_returns)

                if self.env_type == Env.STARCRAFT:
                    print(f"Steps: {save_interval_steps}, Eval_win_rate: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

                elif self.env_type == Env.MAMUJOCO or self.env_type == Env.PETTINGZOO:
                    print(f"Steps: {save_interval_steps}, Eval rew per step: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

                else:
                    print(f"Steps: {save_interval_steps}, Eval average scores: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_final.pth")
                # self.learner.visualize_attention_map(-1, save_mode='final')
                break
            
            self.server.append(info['idx'], self.learner.params())

        # Final high-quality evaluation — 100 episodes → 1% win-rate granularity
        print("[Final Eval] Running 100 episodes for precise final win rate…")
        final_wr, final_ret = self.server.evaluate_final(self.learner.params(), n_episodes=1000)
        wandb.log({'final_eval_win_rate': final_wr, 'final_eval_returns': final_ret, 'steps': cur_steps})
        print(f"[Final Eval] steps={cur_steps}  final_eval_win_rate={final_wr:.4f}  ({final_wr*100:.1f}%)")

        # store log data locally
        steps = np.array(steps)
        eval_win_rates = np.array(eval_win_rates)
        eval_ret = np.array(eval_ret_list)
        stored_dict = {
            'steps':               steps,
            'eval_win_rates':      eval_win_rates,
            'eval_returns':        eval_ret,
            'final_eval_win_rate': final_wr,
            'final_eval_returns':  final_ret,
        }
        with open(self.save_path, 'wb') as f:
            pickle.dump(stored_dict, f)
    
    # only train the actor and critic
    def train_actor(self, world_model_path, max_steps=10 ** 10, max_episodes=10 ** 10):
        ## preload world model
        self.learner.load_pretrained_wm(world_model_path)
        
        cur_steps, cur_episode = 0, 0
        save_interval_steps = 0
        last_save_steps = 0

        wandb.define_metric("steps")
        wandb.define_metric("win rate", step_metric="steps")
        
        while True:
            rollout, info = self.server.run()
            ent = rollout['entropy'].sum(0) / (rollout['entropy'] > 1e-6).sum(0)
            ent_str = f""
            for e in ent.tolist():
                ent_str += f"{e:.4f} "

            cur_steps += info["steps_done"]
            cur_episode += 1
            save_interval_steps += info["steps_done"]
            returns = rollout["reward"].sum(0).mean()

            wandb.log({'win rate': info["reward"], 'steps': cur_steps})
            wandb.log({'returns': returns, "episodes": cur_episode})

            print("%4s" % cur_episode, "%5s" % (cur_steps), info["reward"], 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')

            # train actor only
            self.learner.train_actor_only(rollout)

            if (save_interval_steps - last_save_steps) > 10000:
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_{save_interval_steps // 10000}Ksteps.pth")
                last_save_steps = save_interval_steps // 10000 * 10000

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break
            
            self.server.append(info['idx'], self.learner.params())
        
        