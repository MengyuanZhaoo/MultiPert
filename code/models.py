import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Bernoulli
import os


class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_size*2),
            nn.ReLU(),
            nn.Linear(encoding_size*2, encoding_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, encoding_size*2),
            nn.ReLU(),
            nn.Linear(encoding_size*2, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ZINBVAE(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(ZINBVAE, self).__init__()
        self.encoder_fc1 = nn.Linear(input_size, encoding_size * 2)
        self.encoder_fc2_mu = nn.Linear(encoding_size * 2, encoding_size)
        self.encoder_fc2_logvar = nn.Linear(encoding_size * 2, encoding_size)

        self.decoder_fc1 = nn.Linear(encoding_size, encoding_size * 2)
        self.decoder_fc2_mu = nn.Linear(encoding_size * 2, input_size)
        self.decoder_fc2_theta = nn.Linear(encoding_size * 2, input_size)
        self.decoder_fc2_pi = nn.Linear(encoding_size * 2, input_size)

    def encode(self, x):
        h = nn.functional.relu(self.encoder_fc1(x))
        return self.encoder_fc2_mu(h), self.encoder_fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = nn.functional.relu(self.decoder_fc1(z))
        mu = torch.exp(self.decoder_fc2_mu(h))
        theta = torch.exp(self.decoder_fc2_theta(h))
        pi = torch.sigmoid(self.decoder_fc2_pi(h))
        return mu, theta, pi

    def forward(self, x):
        encoder_mu, logvar = self.encode(x)
        z = self.reparameterize(encoder_mu, logvar)
        decoder_mu, theta, pi = self.decode(z)
        return decoder_mu, theta, pi, encoder_mu, logvar


def zinb_loss(x, mu, theta, pi, mu_kl, logvar_kl):
    x = torch.round(x).clamp(min=0).int()
    nb_dist = NegativeBinomial(total_count=theta, logits=torch.log(mu / theta))
    nb_log_prob = nb_dist.log_prob(x)

    zero_nb = (1 + mu / theta).pow(-theta)
    zero_case = torch.log(pi + (1 - pi) * zero_nb + 1e-8)
    non_zero_case = torch.log(1 - pi + 1e-8) + nb_log_prob

    zinb_nll = -torch.where(x < 1e-8, zero_case, non_zero_case).sum(-1)
    kl_divergence = -0.5 * (1 + logvar_kl - mu_kl.pow(2) - logvar_kl.exp()).sum(-1)

    return (zinb_nll + kl_divergence).mean()


class SharedEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(SharedEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, embedding_size*2),
            nn.ReLU(),
            nn.Linear(embedding_size*2, embedding_size)
        )

    def forward(self, x):
        return self.fc(x)


class PerturbationEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(PerturbationEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, embedding_size*2),
            nn.ReLU(),
            nn.Linear(embedding_size*2, embedding_size)
        )

    def forward(self, x):
        return self.fc(x)

class SharedEncoderRNA(nn.Module):
    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.BatchNorm1d(input_size//2),
            nn.LeakyReLU(0.1),
            nn.Linear(input_size//2, input_size//4),
            nn.BatchNorm1d(input_size//4),
            nn.LeakyReLU(0.1),
            nn.Linear(input_size//4, embedding_size*2),
            nn.BatchNorm1d(embedding_size*2),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size*2, embedding_size)
        )

    def forward(self, x):
        return self.layers(x)


class FusionNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(FusionNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )

    def forward(self, specific, shared):
        combined = torch.cat((specific, shared), dim=1)
        return self.fc(combined)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size//2),
            nn.ReLU(),
            nn.Linear(output_size//2, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class AdversarialNetwork(nn.Module):
    def __init__(self, input_size):
        super(AdversarialNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, rna_dim, perturb_dim, n_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(rna_dim, rna_dim)
        self.kv_proj = nn.Linear(perturb_dim, rna_dim)
        self.attn = nn.MultiheadAttention(rna_dim, n_heads)
        self.LayerNorm = nn.LayerNorm(rna_dim)
        
    def forward(self, rna_emb, perturb_emb):
        B, D = rna_emb.shape

        Q = self.q_proj(rna_emb).unsqueeze(0)
        KV = self.kv_proj(perturb_emb).unsqueeze(0)
        attn_output, attn_weights = self.attn(Q, KV, KV)
        attn_output = attn_output.squeeze(0)
        fused_rna = self.LayerNorm(attn_output + rna_emb)
        
        return fused_rna


class DualFusionAttention(nn.Module):  
    def __init__(self, embed_dim=32):  
        super().__init__()  
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4)  
        self.channel_attn = nn.Sequential(  
            nn.AdaptiveAvgPool1d(1),  
            nn.Flatten(start_dim=1),
            nn.Linear(embed_dim, embed_dim//4),  
            nn.ReLU(),  
            nn.Linear(embed_dim//4, embed_dim),  
            nn.Sigmoid()  
        )  

    def forward(self, cell_embed, perturb_embed):  
        attn_output, _ = self.cross_attn(cell_embed.unsqueeze(0), perturb_embed.unsqueeze(0), perturb_embed.unsqueeze(0))
        attn_output = attn_output.squeeze(0)  
        attn_output_reshaped = attn_output.unsqueeze(-1)
        channel_weights = self.channel_attn(attn_output_reshaped)
        channel_weights = channel_weights.squeeze(-1)
        fused_embed = attn_output * channel_weights  
        return fused_embed  

def initialize_models(train_loader, dim_rna, dim_adt, dim_perturb, device, out_path):
    specific_emb = 32
    rna_ae = ZINBVAE(dim_rna, specific_emb).to(device)
    rna_criterion = zinb_loss

    adt_ae = Autoencoder(dim_adt, specific_emb).to(device)
    adt_criterion = nn.MSELoss()

    rna_optimizer_ae = optim.Adam(rna_ae.parameters(), lr=0.001)
    adt_optimizer_ae = optim.Adam(adt_ae.parameters(), lr=0.001)

    rna_ae_path = os.path.join(out_path, 'rna_ae_pretrained.pth')
    adt_ae_path = os.path.join(out_path, 'adt_ae_pretrained.pth')

    # Pretraining
    if os.path.exists(rna_ae_path) and os.path.exists(adt_ae_path):
        print("Loading pre-trained RNA and ADT autoencoders...")
        rna_ae.load_state_dict(torch.load(rna_ae_path))
        adt_ae.load_state_dict(torch.load(adt_ae_path))
    else:
        print('Pretraining transcriptome-specific encoder...')
        for epoch in range(10):
            rna_total_loss = 0
            for rna, _, _, _, _, _, _, _ in train_loader:
                rna = rna.to(device)
                rna_optimizer_ae.zero_grad()
                decoder_mu, theta, pi, encoder_mu, logvar = rna_ae(rna)
                rna_loss = rna_criterion(rna, decoder_mu, theta, pi, encoder_mu, logvar)
                rna_loss.backward()
                rna_optimizer_ae.step()
                rna_total_loss += rna_loss.item() / len(train_loader)
            print(f'Epoch {epoch + 1}, transcriptome-specific encoder pretraining loss: {rna_total_loss}')

        print('Pretraining proteome-specific encoder...')
        for epoch in range(10):
            adt_total_loss = 0
            for _, adt, _, _, _, _, _, _ in train_loader:
                adt = adt.to(device)
                adt_optimizer_ae.zero_grad()
                adt_output = adt_ae(adt)
                adt_loss = adt_criterion(adt_output, adt)
                adt_loss.backward()
                adt_optimizer_ae.step()
                adt_total_loss += adt_loss.item() / len(train_loader)
            print(f'Epoch {epoch + 1}, proteome-specific encoder pretraining loss: {adt_total_loss}')

        torch.save(rna_ae.state_dict(), rna_ae_path)
        torch.save(adt_ae.state_dict(), adt_ae_path)
        print("Pretrained transcriptome-specific and proteome-specific encoders saved.")


    # Integrative training
    share_emb_size = 32
    fusion_emb_size = 32
    perturb_emb_size = 64
    num_heads = 4

    shared_encoder_rna = SharedEncoder(dim_rna, share_emb_size).to(device)
    shared_encoder_adt = SharedEncoder(dim_adt, share_emb_size).to(device)

    perturb_encoder_rna = PerturbationEncoder(dim_perturb, perturb_emb_size).to(device)
    perturb_encoder_adt = PerturbationEncoder(dim_perturb, perturb_emb_size).to(device)

    fusion_network = FusionNetwork(share_emb_size+share_emb_size, fusion_emb_size).to(device)

    decoder_rna = Decoder(fusion_emb_size+specific_emb, dim_rna).to(device)
    decoder_adt = Decoder(fusion_emb_size+specific_emb, dim_adt).to(device)
    adversarial_network = AdversarialNetwork(share_emb_size).to(device)
    attention_layer = DualFusionAttention(embed_dim=fusion_emb_size+specific_emb).to(device)

    model_list = [rna_ae, shared_encoder_rna, perturb_encoder_rna, decoder_rna,
                  adt_ae, shared_encoder_adt, perturb_encoder_adt, decoder_adt,
                  fusion_network, adversarial_network, attention_layer]

    return model_list