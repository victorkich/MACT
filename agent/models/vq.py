from .vector_quantize_pytorch import FSQ, VectorQuantize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from einops import rearrange

import ipdb


class SimpleVAEAutoEncoder(nn.Module):
    """
    A Variational Autoencoder (VAE) with discrete latent variables modeled by Categorical distributions.
    This is designed to replace SimpleVQAutoEncoder, maintaining a similar interface for MARIE.
    Inspired by VAE approaches like in TWISTER where latents can be categorical.
    """

    def __init__(self,
                 in_dim: int,
                 num_tokens: int,  # K: Number of discrete latent variables per observation
                 num_categories_per_token: int,  # N: Number of categories for each latent variable
                 latent_embedding_dim: int,  # n_z: Embedding dimension for each chosen category
                 hidden_size: int = 256,  # Hidden size for encoder/decoder layers
                 temperature: float = 1.0):  # Temperature for Gumbel-Softmax
        super().__init__()

        self.in_dim = in_dim
        self.num_tokens = num_tokens  # K
        self.num_categories_per_token = num_categories_per_token  # N
        self.latent_embedding_dim = latent_embedding_dim  # n_z
        self.temperature = temperature  # For Gumbel-Softmax

        # Encoder: Maps input observation to logits for the K categorical distributions
        # Output size: num_tokens * num_categories_per_token
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_tokens * num_categories_per_token)
        )

        # Embedding layer for the discrete categories (indices) to feed into the decoder
        # Each of the K tokens (with N categories) will be embedded into latent_embedding_dim
        self.category_embedder = nn.Embedding(num_categories_per_token, latent_embedding_dim)

        # Decoder: Maps embedded latent variables back to observation space
        # Input size: num_tokens * latent_embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(num_tokens * latent_embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_dim)
        )

    @torch.no_grad()
    def encode_logits(self, x: torch.Tensor, *, preprocess: bool = True):
        """
        Returns raw encoder logits without sampling.
        Output shape: (*prefix, K, N)
        """
        if preprocess:
            x = self.preprocess_input(x)
        logits = self._encode_to_logits(x)  # existing private fn
        return logits

    def get_embedded_latents(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the embedded latent states from discrete indices.
        Input: indices of shape (*prefix, num_tokens)
        Output: embedded latents of shape (*prefix, num_tokens, latent_embedding_dim)
        """
        return self.category_embedder(indices)

    def _encode_to_logits(self, x: torch.Tensor) -> torch.Tensor:
        """ Encodes input x to logits for the categorical distributions. """
        orig_shape_prefix = x.shape[:-1]  # All dimensions except the last (feature dim)
        encoder_output = self.encoder(x)
        logits = encoder_output.view(*orig_shape_prefix, self.num_tokens, self.num_categories_per_token)
        return logits

    def encode(self, x: torch.Tensor, should_preprocess: bool = False):
        """
        Encodes observation x into discrete token indices and their embeddings.
        This method is designed to be compatible with how MARIE's world model uses the tokenizer's encode output.
        Output:
            - z_embedded_per_token: Embedded version of the chosen discrete tokens.
                                     Shape: (*x.shape[:-1], num_tokens, latent_embedding_dim)
            - indices: Discrete token indices. Shape: (*x.shape[:-1], num_tokens)
        """
        if should_preprocess:
            x_processed = self.preprocess_input(x)
        else:
            x_processed = x

        logits = self._encode_to_logits(x_processed)

        # For encoding (inference/non-training path), typically use the mode of the distribution (argmax)
        indices = torch.argmax(logits, dim=-1)  # Shape: (*x.shape[:-1], num_tokens)

        # Get the embedded representation of these discrete tokens
        # This corresponds to z_quantized in the original VQ-VAE interface
        z_embedded_per_token = self.category_embedder(indices)
        # Shape: (*x.shape[:-1], num_tokens, latent_embedding_dim)

        return z_embedded_per_token, indices

    def decode(self, indices: torch.Tensor, should_postprocess: bool = False):
        """
        Decodes discrete token indices back into the observation space.
        Input:
            - indices: Discrete token indices. Shape: (*shape_prefix, num_tokens)
        Output:
            - rec: Reconstructed observation. Shape: (*shape_prefix, in_dim)
        """
        orig_shape_prefix = indices.shape[:-1]  # All dimensions except num_tokens

        # Embed the discrete indices
        latents_embedded = self.category_embedder(indices)
        # Shape: (*orig_shape_prefix, num_tokens, latent_embedding_dim)

        # Flatten for decoder input
        decoder_input = latents_embedded.reshape(*orig_shape_prefix, self.num_tokens * self.latent_embedding_dim)

        rec = self.decoder(decoder_input)

        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False):
        """ Encodes and then decodes the input x. """
        # Note: The original SimpleVQAutoEncoder.encode returns z_q (embedded) and indices.
        # We need to ensure this VAE's encode method does the same for compatibility.
        z_embedded, indices = self.encode(x, should_preprocess)
        rec = self.decode(indices, should_postprocess)  # Decode from indices
        return rec

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False):
        """
        Full forward pass for training the VAE.
        Outputs reconstruction, discrete indices, and the KL divergence term for the loss.
        Output:
            - rec: Reconstructed observation.
            - indices: Sampled discrete token indices.
            - kl_scalar: Scalar KL divergence loss component.
        """
        if should_preprocess:
            x_processed = self.preprocess_input(x)
        else:
            x_processed = x

        orig_shape_prefix = x_processed.shape[:-1]

        logits = self._encode_to_logits(x_processed)

        # Sample from the categorical distributions using Gumbel-Softmax
        # hard=True means the output is one-hot, but gradients flow via straight-through estimator
        latents_one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        indices = torch.argmax(latents_one_hot, dim=-1)  # Discrete indices from the Gumbel-Softmax samples

        # Embed the one-hot samples for the decoder
        # This step is differentiable due to Gumbel-Softmax / STE
        # latents_embedded_for_decoder = torch.matmul(latents_one_hot, self.category_embedder.weight)
        # A cleaner way that also works with STE from hard Gumbel-Softmax:
        latents_embedded_for_decoder = self.category_embedder(indices)

        decoder_input = latents_embedded_for_decoder.reshape(*orig_shape_prefix,
                                                             self.num_tokens * self.latent_embedding_dim)
        rec = self.decoder(decoder_input)

        if should_postprocess:
            rec = self.postprocess_output(rec)

        # Calculate KL divergence loss component
        # q_z_k_given_x is Categorical(logits=logits_for_token_k)
        # p_z_k is a uniform Categorical prior
        q_dist = Categorical(logits=logits)  # Shape: (*orig_shape_prefix, num_tokens, num_categories_per_token)

        # Uniform prior for each of the K categorical variables
        prior_probs = torch.full_like(logits, 1.0 / self.num_categories_per_token)
        p_dist = Categorical(probs=prior_probs)

        # KL(q(z|x) || p(z)) = sum_k KL(q(z_k|x) || p(z_k))
        # kl_divergence for Categorical is sum_j q_j * (log q_j - log p_j)
        kl_div_per_token = torch.distributions.kl.kl_divergence(q_dist, p_dist)
        # kl_div_per_token shape: (*orig_shape_prefix, num_tokens)

        # Sum KL over the K tokens, then average over all other dimensions (batch, agents, etc.)
        kl_scalar = kl_div_per_token.sum(dim=-1).mean()

        return rec, indices, kl_scalar

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """ Placeholder for any input preprocessing. """
        return x

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """ Placeholder for any output postprocessing, e.g., clamping. """
        # Example: return y.clamp(-1., 1.)
        return y

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.1, reconstruction_loss_fn=F.mse_loss):
        """
        Computes the VAE loss: reconstruction loss + weighted KL divergence.
        This method is for standalone VAE training if needed, or can be adapted
        by the main training loop (e.g., DreamerLearner).
        """
        # Assuming x is already preprocessed if this is called directly for VAE training
        rec, indices, kl_term = self.forward(x, should_preprocess=False, should_postprocess=False)

        # Reconstruction Loss
        rec_loss = reconstruction_loss_fn(rec, x)

        # Total VAE Loss
        total_loss = rec_loss + kl_weight

        loss_dict = {
            "vae/rec_loss": rec_loss.item(),
            "vae/kl_loss": kl_term.item(),
            "vae/total_loss": total_loss.item(),
        }

        # Optional: calculate "active categories" or other metrics if useful
        # active_categories_count = indices.detach().unique().numel()
        # loss_dict["vae/active_categories_count"] = active_categories_count

        return total_loss, loss_dict


class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_tokens: int, hidden_size: int = 512, **vq_kwargs):
        super().__init__()

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim * num_tokens)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * num_tokens, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, in_dim)
        )

        self.codebook = VectorQuantize(dim=embed_dim, **vq_kwargs)
        return
    
    def encode(self, x, should_preprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices, _ = self.codebook(x)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        z_quantized = self.codebook.get_output_from_indices(indices)
        return z_quantized, indices
        

    def decode(self, indices, should_postprocess: bool = False):
        z_quantized = self.codebook.get_output_from_indices(indices)
        rec = self.decoder(z_quantized)

        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec
    
    @torch.no_grad()
    def encode_decode(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        z_q, indices = self.encode(x, should_preprocess)
        rec = self.decode(indices, should_postprocess)
        return rec

    def forward(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices, commit_loss = self.codebook(x)
        
        x = x.reshape(*shape[:-1], -1)
        rec = self.decoder(x)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        
        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec, indices, commit_loss
     
    def preprocess_input(self, x):
        return x
    
    def postprocess_output(self, y):
        '''
        clamp into [-1, 1]
        '''
        # return y.clamp(-1., 1.)
        return y
    
    def compute_loss(self, x, alpha = 10.):
        out, indices, cmt_loss = self(x, True, True)
        rec_loss = (out - x).abs().mean()
        loss = rec_loss + alpha * cmt_loss
        
        active_rate = indices.detach().unique().numel() / self.codebook.codebook_size * 100
        
        loss_dict = {
            "vq/cmt_loss": cmt_loss.item(),
            "vq/rec_loss": rec_loss.item(),
            "vq/active": active_rate,
        }
        
        return loss, loss_dict

    def encode_logits(self, x, preprocess: bool = False):
        if preprocess:
            x = self.preprocess_input(x)

        shape = x.shape  # [..., in_dim]
        x_enc = self.encoder(x)  # [..., E*H]
        x_enc = rearrange(x_enc, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        # x_enc : [B_flat , H=num_tokens , D]

        # ---------- fetch code-book ----------
        codebook_emb = getattr(self.codebook, 'codebook', None)
        if codebook_emb is None:  # newer vector-quantize
            codebook_emb = self.codebook.embedding.weight
        # codebook_emb is [C,D]  or  [H,C,D]

        # ---------- distance → logits ----------
        if codebook_emb.ndim == 2:  # shared code-book
            dists = torch.cdist(x_enc, codebook_emb)  # [B_flat, H, C]
            logits = -dists.pow(2)

        else:  # separate per head
            B_flat, H, D = x_enc.shape
            _, C, _ = codebook_emb.shape
            # broadcast pairwise ||x - c||^2  over heads
            x_h = x_enc.unsqueeze(2)  # [B_flat,H,1,D]
            c_h = codebook_emb.unsqueeze(0)  # [1,H,C,D]
            logits = -((x_h - c_h).pow(2).sum(-1))  # [B_flat,H,C]

        # ---------- reshape back ----------
        final_shape = (*shape[:-1], self.num_tokens, logits.size(-1))
        return logits.view(final_shape)


class SimpleFSQAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, num_tokens: int, levels, **fsq_kwargs) -> None:
        super().__init__()

        self.num_tokens = num_tokens
        self.levels = levels
        self.embed_dim = len(levels)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, len(levels) * num_tokens)
        )

        self.decoder = nn.Sequential(
            nn.Linear(len(levels) * num_tokens, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, in_dim)
        )

        self.codebook = FSQ(levels, **fsq_kwargs)
        
    def encode(self, x, should_preprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices = self.codebook(x)
        z_quantized = self.codebook.indices_to_codes(indices)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        z_quantized = z_quantized.reshape(*shape[:-1], self.num_tokens, self.embed_dim)
        return z_quantized, indices
        

    def decode(self, indices, should_postprocess: bool = False):
        shape = indices.shape
        indices = rearrange(indices, "... h -> (...) h")

        z_quantized = self.codebook.indices_to_codes(indices)
        z_quantized = rearrange(z_quantized, "... h d -> (...) (h d)")

        rec = self.decoder(z_quantized)

        rec = rec.reshape(*shape[:-1], -1)

        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec
    
    @torch.no_grad()
    def encode_decode(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        z_q, indices = self.encode(x, should_preprocess)
        rec = self.decode(indices, should_postprocess)
        return rec

    def forward(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices = self.codebook(x)
        
        x = x.reshape(*shape[:-1], -1)
        rec = self.decoder(x)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        
        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec, indices


    def preprocess_input(self, x):
        return x
    
    def postprocess_output(self, y):
        # return y.clamp(-1., 1.)
        return y
    
    def compute_loss(self, x):
        out, indices = self(x, True, True)
        loss = (out - x).abs().mean()

        active_rate = indices.detach().unique().numel() / self.codebook.codebook_size * 100
        
        loss_dict = {
            "vq/rec_loss": loss.item(),
            "vq/active": active_rate,
        }
        
        return loss, loss_dict