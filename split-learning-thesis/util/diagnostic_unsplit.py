import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from dataset import DatasetLoader
from all_model.kagn_models import KAGNClientModel
from all_model.pyramid_cnn import PyramidCNNClientModel
from all_model.models import ClientModel
from all_attacks.attack_unsplit import (
    total_variation, l2_loss, normalize_for_client, denormalize,
    compute_psnr, compute_ssim, compute_mse,
)

MAIN_ITERS  = 100
INPUT_ITERS = 20
MODEL_ITERS = 20
SEED        = getattr(Config, "SEED", 42)


def build_fresh_client(in_channels):
    if Config.MODEL_NAME == "KAGN":
        return KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels,
                                degree=Config.DEGREE)
    elif Config.MODEL_NAME == "PyramidCNN":
        return PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels)
    else:
        return ClientModel(in_channels=in_channels)


def load_trained_client(device, in_channels):
    client = build_fresh_client(in_channels).to(device)
    checkpoint_path = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"

    trained = False
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        client.load_state_dict(checkpoint['client_state'])
        trained = True
        print(f"[OK] Loaded trained undefended client weights from: {checkpoint_path}")
    else:
        print(f"[!] No undefended checkpoint found at: {checkpoint_path}")
        print("[!] Proceeding with a RANDOMLY-INITIALIZED undefended client.")
        print("[!] Absolute PSNR/SSIM below will not match your earlier trained-model")
        print("[!] results -- only the A vs B vs C GAP is meaningful here.")

    client.eval()
    return client, trained


def reconstruct(clone_model, smashed_data, input_shape, device, bn_mode='train',
                 lambda_tv=0.1, lambda_l2=1.0, lr_input=0.001, lr_model=0.001):

    batch_size = smashed_data.shape[0]
    target = smashed_data.detach()

    x_pred = torch.full((batch_size, *input_shape), 0.5, device=device, requires_grad=True)

    input_optimizer = optim.Adam([x_pred], lr=lr_input, amsgrad=True)
    model_optimizer = optim.Adam(clone_model.parameters(), lr=lr_model, amsgrad=True)

    for _ in range(MAIN_ITERS):

        clone_model.eval()
        for _ in range(INPUT_ITERS):
            input_optimizer.zero_grad()
            pred = clone_model(normalize_for_client(x_pred, Config.DATASET))
            loss = (F.mse_loss(pred, target)
                    + lambda_tv * total_variation(x_pred)
                    + lambda_l2 * l2_loss(x_pred))
            loss.backward()
            input_optimizer.step()
            with torch.no_grad():
                x_pred.clamp_(0.0, 1.0)

        clone_model.train() if bn_mode == 'train' else clone_model.eval()

        for _ in range(MODEL_ITERS):
            model_optimizer.zero_grad()
            pred = clone_model(normalize_for_client(x_pred.detach(), Config.DATASET))
            loss = F.mse_loss(pred, target)
            loss.backward()
            model_optimizer.step()

    clone_model.eval()
    return x_pred.detach()


def run_condition(name, images, client_model, device, in_channels, input_shape, bn_mode):
    print(f"\n{'='*60}\n  CONDITION {name}\n{'='*60}")

    torch.manual_seed(SEED)
    clone_model = build_fresh_client(in_channels).to(device)

    with torch.no_grad():
        smashed = client_model(images)

    reconstructed = reconstruct(clone_model, smashed, input_shape, device, bn_mode=bn_mode)

    originals_dn = denormalize(images, Config.DATASET)

    psnrs, ssims, mses = [], [], []
    for i in range(images.shape[0]):
        orig = originals_dn[i]
        rec  = reconstructed[i].clamp(0, 1)
        psnrs.append(compute_psnr(orig, rec))
        ssims.append(compute_ssim(orig.unsqueeze(0), rec.unsqueeze(0)))
        mses.append(compute_mse(orig, rec))

    mean_psnr = sum(psnrs) / len(psnrs)
    mean_ssim = sum(ssims) / len(ssims)
    mean_mse  = sum(mses) / len(mses)

    print(f"  PSNR : {mean_psnr:.2f} dB")
    print(f"  SSIM : {mean_ssim:.4f}")
    print(f"  MSE  : {mean_mse:.5f}")

    return {'condition': name, 'psnr': mean_psnr, 'ssim': mean_ssim, 'mse': mean_mse}


if __name__ == "__main__":
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {Config.MODEL_NAME} | Dataset: {Config.DATASET} | Cut layer: {Config.CUT_LAYER}")

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    _, test_loader = dataset.get_loaders()

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    input_shape = (1, 28, 28) if Config.DATASET == 'MNIST' else (3, 32, 32)

    client_model, trained = load_trained_client(device, in_channels)

    images_batch = None
    for imgs, _ in test_loader:
        if imgs.shape[0] >= 4:
            images_batch = imgs[:4].to(device)
            break

    if images_batch is None:
        raise RuntimeError("Could not obtain 4 images from test_loader (batch size < 4).")

    single_image = images_batch[0:1]
    identical_copies = single_image.repeat(4, 1, 1, 1)

    results = []
    results.append(run_condition(
        "A - current (train BN, 4 distinct images)",
        images_batch, client_model, device, in_channels, input_shape, bn_mode='train'))

    results.append(run_condition(
        "B - eval-only BN, 4 distinct images",
        images_batch, client_model, device, in_channels, input_shape, bn_mode='eval'))

    results.append(run_condition(
        "C - train BN, 4 identical copies of one image",
        identical_copies, client_model, device, in_channels, input_shape, bn_mode='train'))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    if not trained:
        print("  [!] TARGET CLIENT WAS RANDOMLY INITIALIZED (no undefended checkpoint found).")
        print("  [!] Absolute PSNR/SSIM are not comparable to trained-model results.")
    print(f"{'Condition':<45} {'PSNR':>8} {'SSIM':>8} {'MSE':>10}")
    for r in results:
        print(f"{r['condition']:<45} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {r['mse']:>10.5f}")

    print("\nInterpretation:")
    print("  A vs B large gap  -> live BatchNorm batch statistics are destabilizing reconstruction")
    print("  A vs C large gap  -> cross-image competition for shared clone capacity is the issue")
    print("  A, B, C all close -> representation itself is hard to invert at this cut layer")