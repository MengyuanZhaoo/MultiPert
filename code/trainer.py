import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from utils import set_seed

def train_and_validate(model_list,train_loader, val_loader, device, out_path, epochs=1000, patience=20):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0
    best_model_states = None

    discriminator_losses = []
    generator_losses = []

    rna_ae, shared_encoder_rna, perturb_encoder_rna, decoder_rna, \
    adt_ae, shared_encoder_adt, perturb_encoder_adt, decoder_adt, \
    fusion_network, adversarial_network, attention_layer= model_list

    generator_params = list(shared_encoder_rna.parameters()) + \
                       list(shared_encoder_adt.parameters()) + \
                       list(rna_ae.parameters()) + \
                       list(adt_ae.parameters()) + \
                       list(perturb_encoder_rna.parameters()) + \
                       list(perturb_encoder_adt.parameters()) + \
                       list(decoder_rna.parameters()) + \
                       list(decoder_adt.parameters()) + \
                       list(fusion_network.parameters()) + \
                       list(attention_layer.parameters())
    discriminator_params = list(adversarial_network.parameters())
    
    optimizer_g = optim.Adam(generator_params, lr=0.001)
    optimizer_d = optim.Adam(discriminator_params, lr=0.002)
    criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    for epoch in range(epochs):
        train_loss = 0
        epoch_d_loss = 0
        epoch_g_loss = 0

        for model in model_list:
            model.train()

        for rna, adt, perturb_info_batch, perturbed_rna, perturbed_adt, perturb_emb, _, _ in train_loader:
            rna, adt, perturbed_rna, perturbed_adt, perturb_emb = rna.to(device), adt.to(device), perturbed_rna.to(
                device), perturbed_adt.to(device), perturb_emb.to(device)

            rna_specific, _ = rna_ae.encode(rna)
            rna_shared = shared_encoder_rna(rna)

            adt_specific = adt_ae.encoder(adt)
            adt_shared = shared_encoder_adt(adt)

            fused = fusion_network(rna_shared, adt_shared)

            rna_fused = torch.cat((rna_specific, fused), dim=1)
            perturbation_rna = perturb_encoder_rna(perturb_emb)

            rna_combined = attention_layer(rna_fused, perturbation_rna)
            final_rna = decoder_rna(rna_combined)

            adt_fused = torch.cat((adt_specific, fused), dim=1)
            perturbation_adt = perturb_encoder_adt(perturb_emb)

            adt_combined = attention_layer(adt_fused, perturbation_adt)
            final_adt = decoder_adt(adt_combined)

            rna_loss_fusion = criterion(final_rna, perturbed_rna)
            adt_loss_fusion = criterion(final_adt, perturbed_adt)
            reconstruction_loss = rna_loss_fusion + adt_loss_fusion

            optimizer_d.zero_grad()
            for param in generator_params:
                param.requires_grad = False
            rna_shared_labels = torch.ones((rna.shape[0], 1)).to(device)
            adt_shared_labels = torch.zeros((adt.shape[0], 1)).to(device)
            rna_shared_pred = adversarial_network(rna_shared)
            adt_shared_pred = adversarial_network(adt_shared)
            discriminator_loss = adversarial_criterion(rna_shared_pred, rna_shared_labels) + \
                                 adversarial_criterion(adt_shared_pred, adt_shared_labels)
            discriminator_loss.backward(retain_graph=True)
            optimizer_d.step()
            epoch_d_loss += discriminator_loss.item()

            optimizer_g.zero_grad()
            for param in generator_params:
                param.requires_grad = True
            for param in discriminator_params:
                param.requires_grad = False
            rna_shared_pred = adversarial_network(rna_shared)
            adt_shared_pred = adversarial_network(adt_shared)
            reverse_rna_labels = torch.ones((rna.shape[0], 1), device=device) * 0.5
            reverse_adt_labels = torch.ones((adt.shape[0], 1), device=device) * 0.5
            generator_adv_loss = adversarial_criterion(rna_shared_pred, reverse_rna_labels) + \
                                 adversarial_criterion(adt_shared_pred, reverse_adt_labels)
            total_generator_loss = reconstruction_loss + generator_adv_loss
            total_generator_loss.backward()
            optimizer_g.step()
            
            for param in discriminator_params:
                param.requires_grad = True

            total_loss = total_generator_loss+discriminator_loss
            train_loss += total_loss.item()
            epoch_g_loss += generator_adv_loss.item()

        train_losses.append(train_loss / len(train_loader))
        discriminator_losses.append(epoch_d_loss / len(train_loader))
        generator_losses.append(epoch_g_loss / len(train_loader))

        # Validataion
        val_loss = 0
        for model in model_list:
            model.eval()

        with torch.no_grad():
            for rna, adt, perturb_info_batch, perturbed_rna, perturbed_adt, perturb_emb, _, _ in val_loader:
                rna, adt, perturbed_rna, perturbed_adt, perturb_emb = rna.to(device), adt.to(device), perturbed_rna.to(
                    device), perturbed_adt.to(device), perturb_emb.to(device)

                rna_specific, _ = rna_ae.encode(rna)
                rna_shared = shared_encoder_rna(rna)

                adt_specific = adt_ae.encoder(adt)
                adt_shared = shared_encoder_adt(adt)

                fused = fusion_network(rna_shared, adt_shared)

                rna_fused = torch.cat((rna_specific, fused), dim=1)
                perturbation_rna = perturb_encoder_rna(perturb_emb)

                rna_combined = attention_layer(rna_fused, perturbation_rna)
                final_rna = decoder_rna(rna_combined)

                adt_fused = torch.cat((adt_specific, fused), dim=1)
                perturbation_adt = perturb_encoder_adt(perturb_emb)

                adt_combined = attention_layer(adt_fused, perturbation_adt)
                final_adt = decoder_adt(adt_combined)

                rna_loss_fusion = criterion(final_rna, perturbed_rna)
                adt_loss_fusion = criterion(final_adt, perturbed_adt)

                rna_shared_labels = torch.ones((rna.shape[0], 1)).to(device)
                adt_shared_labels = torch.zeros((adt.shape[0], 1)).to(device)
                rna_shared_pred = adversarial_network(rna_shared)
                adt_shared_pred = adversarial_network(adt_shared)
                discriminator_loss = adversarial_criterion(rna_shared_pred, rna_shared_labels) + \
                                        adversarial_criterion(adt_shared_pred, adt_shared_labels)

                reverse_rna_labels = torch.ones((rna.shape[0], 1), device=device) * 0.5
                reverse_adt_labels = torch.ones((adt.shape[0], 1), device=device) * 0.5
                generator_adv_loss = adversarial_criterion(rna_shared_pred, reverse_rna_labels) + \
                                    adversarial_criterion(adt_shared_pred, reverse_adt_labels)
                val_loss += (rna_loss_fusion + adt_loss_fusion+discriminator_loss+generator_adv_loss).item()

        val_losses.append(val_loss / len(val_loader))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            counter = 0
            best_model_states = [model.state_dict() for model in model_list]
        else:
            counter += 1

        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        if (epoch + 1) % 10 == 0:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'{current_time} - Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

    if best_model_states is not None:
        for i, model in enumerate(model_list):
            model.load_state_dict(best_model_states[i])
            torch.save(model.state_dict(), out_path+'/model_best_'+str(i)+'.pth')
        print("Best model saved.")

    for i, model in enumerate(model_list):
        torch.save(model.state_dict(), out_path+'/model_last_'+str(i)+'.pth')
    print("Last model saved.")

    return train_losses, val_losses, discriminator_losses, generator_losses

