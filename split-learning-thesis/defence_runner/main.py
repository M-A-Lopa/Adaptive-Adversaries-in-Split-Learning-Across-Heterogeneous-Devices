import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from dataset import DatasetLoader
from all_split_learning.split_learning import SplitLearningTrainer
from all_attacks.attack_unsplit import UnSplitAttack
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.ae_decoder_attack import run_ae_decoder_attack, save_ae_attack_visualization
from all_attacks.fsha_attack import FSHAAttack
from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders
from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack

def plot_results(train_losses, train_accuracies, test_accuracies, dataset_name):
    
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.set_title('Training Loss', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accuracies, 'b-', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, test_accuracies,  'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Model Accuracy', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{Config.MODEL_NAME} Split Learning — {dataset_name}', fontsize=14)
    plt.tight_layout()

    save_path = f"{Config.RESULTS_DIR}/training_curves_{dataset_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    print(f"Training curves saved → {save_path}")


if __name__ == "__main__":

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    
    if Config.MODEL_NAME == "KAGN":
        client_model = KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE).to(device)
        server_model = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES, in_channels=in_channels, degree=Config.DEGREE).to(device)
    elif Config.MODEL_NAME == "PyramidCNN":
        client_model = PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels).to(device)
        server_model = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES,in_channels=in_channels).to(device)
    else:
        client_model = ClientModel(in_channels=in_channels).to(device)
        server_model = ServerModel(num_classes=Config.NUM_CLASSES).to(device)
        
        
    def build_fresh_client(): #Used by Unsplit
        if Config.MODEL_NAME == "KAGN":
            return KAGNClientModel(cut_layer=Config.CUT_LAYER,
                                   in_channels=in_channels, degree=3)
        elif Config.MODEL_NAME == "PyramidCNN":
            return PyramidCNNClientModel(cut_layer=Config.CUT_LAYER,
                                         in_channels=in_channels)
        else:
            return ClientModel(in_channels=in_channels)

    def build_fresh_split(num_classes): #Used by Label Leakage and VILLAIN
        if Config.MODEL_NAME == "KAGN":
            fresh_client = KAGNClientModel(cut_layer=Config.CUT_LAYER,
                                           in_channels=in_channels, degree=Config.DEGREE)
            fresh_server = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes,
                                           in_channels=in_channels, degree=Config.DEGREE)
        elif Config.MODEL_NAME == "PyramidCNN":
            fresh_client = PyramidCNNClientModel(cut_layer=Config.CUT_LAYER,
                                                 in_channels=in_channels)
            fresh_server = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes,
                                                 in_channels=in_channels)
        else:
            fresh_client = ClientModel(in_channels=in_channels)
            fresh_server = ServerModel(num_classes=num_classes)
        return fresh_client, fresh_server

    checkpoint_path = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"\n[✓] Found existing trained weights for {Config.MODEL_NAME} at: {checkpoint_path}")
        print("Skipping training phase. Loading weights into network structures...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        client_model.load_state_dict(checkpoint['client_state'])
        server_model.load_state_dict(checkpoint['server_state'])
    else:
        print(f"\n[!] No local checkpoint found at: {checkpoint_path}")
        print(f"Initiating {Config.MODEL_NAME} Split Learning training pipeline...")
        
        trainer = SplitLearningTrainer(client_model=client_model, server_model=server_model, train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        trainer.save_results()

    if Config.RUN_ATTACK:

        MAX_IMAGES  = 32
        ITERATIONS  = 1000

        HIJACK_EPOCHS    = 5
        CRITIC_ITERS     = 5

        LEAKAGE_EPOCHS   = 5
        POSITIVE_CLASS   = 0
        POSITIVE_RATIO   = 0.1
        LEAKAGE_BATCH    = 128
        LEAKAGE_LR       = 1e-4

        WARMUP_EPOCHS    = 5
        INFERENCE_EPOCHS = 5
        INJECTION_EPOCHS = 10
        VILLAIN_BATCH    = 128
        TARGET_LABEL     = 0
        TRIGGER_BETA     = 1.0
        TRIGGER_FRACTION = 0.5
        DROPOUT_KEEP     = 0.75
        GAMMA_LOW        = 0.6
        GAMMA_HIGH       = 1.2
        POISON_RATE      = 0.01
        CANDIDATES       = 14

        BACKDOOR_TARGET_LABEL     = 0
        BACKDOOR_POISON_RATE      = 0.05
        BACKDOOR_PATCH_SIZE       = 4
        BACKDOOR_TRIGGER_VALUE    = 1.0
        BACKDOOR_TRAIN_EPOCHS     = 10
        BACKDOOR_SURROGATE_EPOCHS = 5   # server mode only

        if Config.DATASET == 'CIFAR10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
            std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
        else:
            mean = torch.tensor([0.1307]).view(1,1,1,1).to(device)
            std  = torch.tensor([0.3081]).view(1,1,1,1).to(device)

        all_results = {}

        # Attack 1: White-Box rMSE
        print("\n" + "="*60)
        print(f"  WHITE-BOX ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        attacker = WhiteBoxInversionAttack(client_model=client_model,
                                           dataset=Config.DATASET,
                                           iterations=ITERATIONS, lr=1e-2)
        tracker = AttackMetricsTracker()
        images_processed = 0

        for inputs, _ in test_loader:
            if images_processed >= MAX_IMAGES:
                break
            inputs    = inputs.to(device)
            remaining = MAX_IMAGES - images_processed
            inputs    = inputs[:remaining]

            print(f"\n  Inverting {inputs.shape[0]} image(s) "
                  f"[{images_processed + inputs.shape[0]}/{MAX_IMAGES}]...")

            with torch.no_grad():
                target_smashed = client_model(inputs)
            reconstructed = attacker.reconstruct(target_smashed, inputs.shape)

            inputs_denorm = torch.clamp(inputs * std + mean, 0, 1)
            tracker.log_batch(inputs_denorm, reconstructed)
            images_processed += inputs.shape[0]

        wb_summary = tracker.get_summary()
        all_results['WhiteBox'] = wb_summary
        print(f"\n  PSNR: {wb_summary['mean_psnr']:.2f} dB | "
              f"SSIM: {wb_summary['mean_ssim']:.4f}")

        pd.DataFrame([wb_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_whitebox_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        #Attack 2: UnSplit (coordinate gradient descent, data-oblivious)
        print("\n" + "="*60)
        print(f"  UNSPLIT ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        unsplit_attacker = UnSplitAttack(
            client_model=client_model,
            in_channels=in_channels,
            clone_builder=build_fresh_client
        )
        unsplit_summary = unsplit_attacker.run_attack(
            test_loader, num_batches=MAX_IMAGES // 32 or 1,
            inversion_steps=ITERATIONS
        )
        all_results['UnSplit'] = unsplit_summary
        print(f"\n  PSNR: {unsplit_summary['psnr']:.2f} dB | "
              f"SSIM: {unsplit_summary['ssim']:.4f}")

        pd.DataFrame([unsplit_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_unsplit_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        #Attack 3: AE Decoder (model-based inversion)
        print("\n" + "="*60)
        print(f"  AE DECODER ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        ae_summary, ae_orig, ae_recon = run_ae_decoder_attack(
            client_model=client_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            dataset=Config.DATASET,
            preprocess_fn=None,
            ae_epochs=50,
            label=f'{Config.MODEL_NAME} AE Decoder'
        )
        all_results['AE_Decoder'] = ae_summary
        pd.DataFrame([ae_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_ae_decoder_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        if len(ae_orig) > 0:
            save_ae_attack_visualization(
                originals=ae_orig, reconstructed=ae_recon,
                baseline_summary=wb_summary,     
                defense_summary=ae_summary,
                defense_name=Config.MODEL_NAME,
                dataset=Config.DATASET,
                results_dir=Config.RESULTS_DIR
            )

        #Attack 4: FSHA 
        print("\n" + "="*60)
        print(f"  FSHA ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        fsha_client = build_fresh_client().to(device)

        fsha_attacker = FSHAAttack(
            client_model=fsha_client,
            in_channels=in_channels,
            dataset=Config.DATASET,
            pilot_builder=build_fresh_client,
            critic_iters=CRITIC_ITERS
        )
        fsha_attacker.hijack(train_loader, test_loader, epochs=HIJACK_EPOCHS)
        fsha_summary = fsha_attacker.reconstruct(train_loader, num_images=MAX_IMAGES)
        all_results['FSHA'] = fsha_summary

        fsha_source = f"{Config.RESULTS_DIR}/fsha_no_defense.png"
        fsha_target = f"{Config.RESULTS_DIR}/fsha_{Config.MODEL_NAME.lower()}_{Config.DATASET}.png"
        if os.path.exists(fsha_source):
            if os.path.exists(fsha_target):
                os.remove(fsha_target)
            os.rename(fsha_source, fsha_target)

        pd.DataFrame([fsha_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_fsha_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        #Attack 5: Gradient-Norm Label Leakage 
        print("\n" + "="*60)
        print(f"  LABEL LEAKAGE ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        binary_train_loader, binary_test_loader = build_binary_split_loaders(
            train_loader.dataset,
            target_class=POSITIVE_CLASS,
            positive_ratio=POSITIVE_RATIO,
            batch_size=LEAKAGE_BATCH
        )

        leakage_client, leakage_server = build_fresh_split(num_classes=1)

        leakage_attacker = GradientNormLabelLeakageAttack(
            client_model=leakage_client,
            server_model=leakage_server,
            dataset=Config.DATASET,
            target_class=POSITIVE_CLASS,
            learning_rate=LEAKAGE_LR
        )
        leakage_summary = leakage_attacker.run(
            binary_train_loader, binary_test_loader, epochs=LEAKAGE_EPOCHS
        )
        leakage_attacker.save_visualization(
            tag=f"{Config.MODEL_NAME.lower()}_{Config.DATASET}"
        )
        all_results['LabelLeakage'] = leakage_summary

        pd.DataFrame([leakage_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_label_leakage_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )
        pd.DataFrame(leakage_attacker.history).to_csv(
            f"{Config.RESULTS_DIR}/label_leakage_batches_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        #Attack 6: VILLAIN Client-Side Trigger Backdoor 
        print("\n" + "="*60)
        print(f"  VILLAIN BACKDOOR ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        indexed_loader = build_indexed_loader(
            train_loader.dataset, batch_size=VILLAIN_BATCH, shuffle=True
        )

        villain_client, villain_server = build_fresh_split(num_classes=Config.NUM_CLASSES)

        villain_attacker = VILLAINBackdoorAttack(
            client_model=villain_client,
            server_model=villain_server,
            base_dataset=train_loader.dataset,
            dataset=Config.DATASET,
            num_classes=Config.NUM_CLASSES,
            target_label=TARGET_LABEL,
            beta=TRIGGER_BETA,
            trigger_fraction=TRIGGER_FRACTION,
            dropout_keep=DROPOUT_KEEP,
            gamma_low=GAMMA_LOW,
            gamma_high=GAMMA_HIGH,
            poison_rate=POISON_RATE,
            candidates_per_batch=CANDIDATES
        )

        villain_attacker.warmup(indexed_loader, epochs=WARMUP_EPOCHS)
        villain_baseline, _ = villain_attacker.evaluate(test_loader)
        print(f"  Clean data accuracy before attack: {villain_baseline:.2f}%")

        villain_attacker.infer_labels(indexed_loader, epochs=INFERENCE_EPOCHS)
        villain_attacker.fabricate_trigger(indexed_loader)
        villain_attacker.inject_backdoor(indexed_loader, test_loader, epochs=INJECTION_EPOCHS)

        villain_summary = villain_attacker.summarise(
            test_loader=test_loader, clean_baseline=villain_baseline
        )
        villain_attacker.save_visualization(
            tag=f"{Config.MODEL_NAME.lower()}_{Config.DATASET}"
        )
        all_results['VILLAIN'] = villain_summary

        pd.DataFrame([villain_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_villain_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )
        pd.DataFrame(villain_attacker.history).to_csv(
            f"{Config.RESULTS_DIR}/villain_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        #Attack 7: Backdoor Poisoning -- Client-Side 
        print("\n" + "="*60)
        print(f"  BACKDOOR POISONING ATTACK (CLIENT-SIDE) — {Config.MODEL_NAME}")
        print("="*60)

        backdoor_c_client, backdoor_c_server = build_fresh_split(num_classes=Config.NUM_CLASSES)

        backdoor_client_attacker = BackdoorPoisonAttack(
            client_model=backdoor_c_client,
            server_model=backdoor_c_server,
            base_dataset=train_loader.dataset,
            dataset=Config.DATASET,
            num_classes=Config.NUM_CLASSES,
            mode='client',
            target_label=BACKDOOR_TARGET_LABEL,
            poison_rate=BACKDOOR_POISON_RATE,
            patch_size=BACKDOOR_PATCH_SIZE,
            trigger_value=BACKDOOR_TRIGGER_VALUE,
            model_tag=f"{Config.MODEL_NAME.lower()}_sl",
        )

        backdoor_client_attacker.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
        backdoor_client_summary = backdoor_client_attacker.summarise()
        backdoor_client_attacker.save_visualization(
            tag=f"{Config.MODEL_NAME.lower()}_{Config.DATASET}"
        )
        all_results['BackdoorPoison_Client'] = backdoor_client_summary

        pd.DataFrame([backdoor_client_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_backdoor_poison_client_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )
        pd.DataFrame(backdoor_client_attacker.history).to_csv(
            f"{Config.RESULTS_DIR}/backdoor_poison_client_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        #Attack 8: Backdoor Poisoning -- Server-Side
        print("\n" + "="*60)
        print(f"  BACKDOOR POISONING ATTACK (SERVER-SIDE) — {Config.MODEL_NAME}")
        print("="*60)

        backdoor_s_client, backdoor_s_server = build_fresh_split(num_classes=Config.NUM_CLASSES)

        backdoor_server_attacker = BackdoorPoisonAttack(
            client_model=backdoor_s_client,
            server_model=backdoor_s_server,
            base_dataset=train_loader.dataset,
            dataset=Config.DATASET,
            num_classes=Config.NUM_CLASSES,
            mode='server',
            target_label=BACKDOOR_TARGET_LABEL,
            poison_rate=BACKDOOR_POISON_RATE,
            patch_size=BACKDOOR_PATCH_SIZE,
            trigger_value=BACKDOOR_TRIGGER_VALUE,
            surrogate_builder=build_fresh_client,   # attacker's own client-arch clone
            model_tag=f"{Config.MODEL_NAME.lower()}_sl",
        )

        backdoor_server_attacker.pretrain_server_backdoor(
            test_loader, epochs=BACKDOOR_SURROGATE_EPOCHS
        )
        backdoor_server_attacker.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
        backdoor_server_summary = backdoor_server_attacker.summarise()
        backdoor_server_attacker.save_visualization(
            tag=f"{Config.MODEL_NAME.lower()}_{Config.DATASET}"
        )
        all_results['BackdoorPoison_Server'] = backdoor_server_summary

        pd.DataFrame([backdoor_server_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_backdoor_poison_server_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )
        pd.DataFrame(backdoor_server_attacker.history).to_csv(
            f"{Config.RESULTS_DIR}/backdoor_poison_server_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        # ── Combined summary table ────────────────────────────────
        blank = "—"

        print("\n" + "="*94)
        print(f"   ALL ATTACKS SUMMARY — {Config.MODEL_NAME} on {Config.DATASET}")
        print("="*94)
        print(f"{'Attack':<20} {'PSNR (dB)':>11} {'SSIM':>9} {'MSE':>9} "
              f"{'Leak AUC':>10} {'LIA (%)':>9} {'ASR (%)':>9} {'CDA (%)':>9}")
        print("-"*94)
        print(f"  {'White-Box':<18} {wb_summary['mean_psnr']:>11.2f} {wb_summary['mean_ssim']:>9.4f} "
              f"{blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
        print(f"  {'UnSplit':<18} {unsplit_summary['psnr']:>11.2f} {unsplit_summary['ssim']:>9.4f} "
              f"{blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
        print(f"  {'AE Decoder':<18} {ae_summary['mean_psnr']:>11.2f} {ae_summary['mean_ssim']:>9.4f} "
              f"{blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
        print(f"  {'FSHA':<18} {fsha_summary['psnr']:>11.2f} {fsha_summary['ssim']:>9.4f} "
              f"{fsha_summary['mse']:>9.5f} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
        print(f"  {'Label Leakage':<18} {blank:>11} {blank:>9} {blank:>9} "
              f"{leakage_summary['q95_norm_leak_auc_cut']:>10.4f} {blank:>9} {blank:>9} {blank:>9}")
        print(f"  {'VILLAIN':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} "
              f"{villain_summary['lia']:>9.2f} {villain_summary['asr']:>9.2f} {villain_summary['cda']:>9.2f}")
        print(f"  {'Backdoor(Client)':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} "
              f"{blank:>9} {backdoor_client_summary['asr']:>9.2f} {backdoor_client_summary['cda']:>9.2f}")
        print(f"  {'Backdoor(Server)':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} "
              f"{blank:>9} {backdoor_server_summary['asr']:>9.2f} {backdoor_server_summary['cda']:>9.2f}")
        print("="*94)

        print("\n" + "-"*94)
        print("   LABEL LEAKAGE DETAIL (95% quantile leak AUC over batches)")
        print("-"*94)
        print(f"  {'Norm (cut layer)':<28} {leakage_summary['q95_norm_leak_auc_cut']:>10.4f}")
        print(f"  {'Cosine (cut layer)':<28} {leakage_summary['q95_cosine_leak_auc_cut']:>10.4f}")
        print(f"  {'Norm (first layer)':<28} {leakage_summary['q95_norm_leak_auc_first']:>10.4f}")
        print(f"  {'Cosine (first layer)':<28} {leakage_summary['q95_cosine_leak_auc_first']:>10.4f}")
        print(f"  {'Majority counting accuracy':<28} {leakage_summary['q95_majority_accuracy_cut']:>10.4f}")
        print("-"*94)

        combined_path = f"{Config.RESULTS_DIR}/all_attacks_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv"
        pd.DataFrame({
            'attack': ['WhiteBox', 'UnSplit', 'AE_Decoder', 'FSHA', 'LabelLeakage', 'VILLAIN',
                       'BackdoorPoison_Client', 'BackdoorPoison_Server'],
            'psnr': [wb_summary['mean_psnr'], unsplit_summary['psnr'], ae_summary['mean_psnr'],
                     fsha_summary['psnr'], None, None, None, None],
            'ssim': [wb_summary['mean_ssim'], unsplit_summary['ssim'], ae_summary['mean_ssim'],
                     fsha_summary['ssim'], None, None, None, None],
            'mse': [None, None, None, fsha_summary['mse'], None, None, None, None],
            'norm_leak_auc_cut': [None, None, None, None,
                                  leakage_summary['q95_norm_leak_auc_cut'], None, None, None],
            'cosine_leak_auc_cut': [None, None, None, None,
                                    leakage_summary['q95_cosine_leak_auc_cut'], None, None, None],
            'norm_leak_auc_first': [None, None, None, None,
                                    leakage_summary['q95_norm_leak_auc_first'], None, None, None],
            'cosine_leak_auc_first': [None, None, None, None,
                                      leakage_summary['q95_cosine_leak_auc_first'], None, None, None],
            'majority_accuracy': [None, None, None, None,
                                  leakage_summary['q95_majority_accuracy_cut'], None, None, None],
            'lia': [None, None, None, None, None, villain_summary['lia'], None, None],
            'asr': [None, None, None, None, None, villain_summary['asr'],
                    backdoor_client_summary['asr'], backdoor_server_summary['asr']],
            'cda': [None, None, None, None, None, villain_summary['cda'],
                    backdoor_client_summary['cda'], backdoor_server_summary['cda']]
        }).to_csv(combined_path, index=False)
        print(f"\n  Combined results saved → {combined_path}")