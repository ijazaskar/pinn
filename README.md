"""
NMC Cathode Young's Modulus - Physics-Informed Neural Network (PINN)
=====================================================================
Study : Molecular dynamics simulations of NMC cathode materials
        with varying composition (Ni/Mn/Co ratio) and state of charge (SoC).
        File naming: {composition}_{SoC}.lammps  e.g. 111_0.5.lammps

Data directory: D:\\research\\Manuscripts\\paper_7_tahir\\Code\\data_complete\\

Physics embedded:
  1. Hooke's law linearity  : d2(sigma)/d(epsilon)^2 ~ 0  in elastic regime
  2. Zero-increment BC       : delta_sigma(epsilon=0) = 0
  3. Composition smoothness  : dSigma/d(x_Ni), dSigma/d(x_Mn) penalised

Key output: E(composition, SoC) via automatic differentiation of trained net.

Requirements:
    pip install torch numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(r"D:\research\Manuscripts\paper_7_tahir\Code\data_complete")
OUTPUT_DIR = Path(r"D:\research\Manuscripts\paper_7_tahir\Code")

ELASTIC_STRAIN_LIMIT = 0.05
N_EPOCHS      = 15_000
LR            = 5e-4
N_HIDDEN      = 5
N_NEURONS     = 128
LAMBDA_DATA   = 1.0
LAMBDA_HOOKE  = 0.05
LAMBDA_BC     = 0.2
LAMBDA_SMOOTH = 0.005
PRINT_EVERY   = 1_000


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_lammps_file(filepath):
    """
    Parse one LAMMPS output file.
    Detects the thermo block by looking for a header line that contains
    both 'Step' and 'Lx', then reads numeric rows below it.
    Returns DataFrame(step, stress_xx, Lx, strain) or None.
    """
    rows = []
    in_data = False

    with open(filepath, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()

            # Header detection: must contain Step, Press, and Lx
            if not in_data and "Step" in s and "Press" in s and "Lx" in s:
                in_data = True
                continue

            if in_data:
                if not s or s.startswith("Loop") or s.startswith("Performance"):
                    in_data = False
                    continue
                parts = s.split()
                if len(parts) < 17:
                    continue
                try:
                    step      = int(parts[0])
                    stress_xx = float(parts[5])   # v_p2 column
                    Lx        = float(parts[16])  # box length x
                    rows.append({"step": step, "stress_xx": stress_xx, "Lx": Lx})
                except (ValueError, IndexError):
                    continue

    if len(rows) < 5:
        return None

    df  = pd.DataFrame(rows)
    Lx0 = df["Lx"].iloc[0]
    df["strain"] = (df["Lx"] - Lx0) / Lx0
    return df


def parse_name(stem):
    """
    '111_0.5' -> {x_Ni:0.333, x_Mn:0.333, x_Co:0.333, SoC:0.5, comp:'111'}
    '811_1'   -> {x_Ni:0.8,   x_Mn:0.1,   x_Co:0.1,   SoC:1.0, comp:'811'}
    Returns None on any parse failure.
    """
    idx = stem.find("_")
    if idx == -1:
        return None

    comp_str = stem[:idx]
    soc_str  = stem[idx + 1:]

    if len(comp_str) != 3 or not comp_str.isdigit():
        return None
    try:
        soc = float(soc_str)
    except ValueError:
        return None

    ni, mn, co = int(comp_str[0]), int(comp_str[1]), int(comp_str[2])
    total = (ni + mn + co) or 1

    return {"x_Ni": ni/total, "x_Mn": mn/total, "x_Co": co/total,
            "SoC": soc, "comp": comp_str}


def build_dataset(data_dir):
    files = sorted(data_dir.glob("*.lammps"))
    print(f"Found {len(files)} LAMMPS files in {data_dir}")
    all_dfs = []

    for fp in files:
        info = parse_name(fp.stem)
        if info is None:
            print(f"  Skipping (bad name)  : {fp.name}")
            continue
        df = parse_lammps_file(fp)
        if df is None:
            print(f"  Skipping (no data)   : {fp.name}")
            continue
        for k, v in info.items():
            df[k] = v
        df["filename"] = fp.name
        all_dfs.append(df)
        print(f"  Parsed : {fp.name}  ->  {len(df)} rows  "
              f"Ni={info['x_Ni']:.2f}  SoC={info['SoC']}")

    if not all_dfs:
        raise RuntimeError(f"No valid data parsed from {data_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows from {len(all_dfs)} simulations\n")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. PINN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class PINN(nn.Module):
    """
    Inputs  (5) : [epsilon, x_Ni, x_Mn, x_Co, SoC]
    Output  (1) : delta_sigma_xx relative to strain=0 (GPa)
    tanh activations ensure smooth, differentiable output.
    """

    def __init__(self, n_hidden=5, n_neurons=128):
        super().__init__()
        layers = [nn.Linear(5, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PHYSICS RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────

def physics_residuals(model, x_col, eps_mean, eps_scale, eps0_scaled):
    """
    Compute three physics-based loss terms via automatic differentiation.
    x_col : (N, 5) standardized tensor on the correct device.
    eps_mean, eps_scale : strain StandardScaler parameters.
    eps0_scaled : standardized coordinate representing physical strain=0.
    """
    # Fresh copy with grad enabled so graph is built from scratch each call
    x = x_col.detach().clone().requires_grad_(True)
    sigma = model(x)                         # (N, 1)
    ones  = torch.ones_like(sigma)

    # First derivative dσ/d(all inputs)
    g1 = torch.autograd.grad(sigma, x,
                              grad_outputs=ones,
                              create_graph=True,
                              retain_graph=True)[0]   # (N, 5)

    dsigma_deps = g1[:, 0:1]                 # (N, 1) — slope w.r.t. epsilon

    # Second derivative d²σ/dε²
    g2 = torch.autograd.grad(dsigma_deps, x,
                              grad_outputs=torch.ones_like(dsigma_deps),
                              create_graph=True,
                              retain_graph=True)[0]   # (N, 5)
    d2sigma_deps2 = g2[:, 0:1]              # (N, 1)

    # R1 — Hooke linearity: d²σ/dε² -> 0 in elastic region
    # Convert standardized strain back to physical strain before masking.
    eps_raw  = x[:, 0:1].detach() * eps_scale + eps_mean
    mask     = (eps_raw.abs() < 0.02).float()
    r_hooke  = (mask * d2sigma_deps2).pow(2).mean()

    # R2 — Zero-stress BC: sigma(epsilon=0) = 0
    x_bc       = x.detach().clone()
    # Standardized zero is mean strain; eps0_scaled is physical strain zero.
    x_bc[:, 0] = eps0_scaled
    sigma_bc   = model(x_bc)
    r_bc       = sigma_bc.pow(2).mean()

    # R3 — Composition smoothness (detach to avoid retaining graph too long)
    dsigma_dNi = g1[:, 1:2]
    dsigma_dMn = g1[:, 2:3]
    r_smooth   = (dsigma_dNi.pow(2) + dsigma_dMn.pow(2)).mean()

    return {"r_hooke": r_hooke, "r_bc": r_bc, "r_smooth": r_smooth}


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_pinn(model, X_train, y_train, X_phys,
               eps_mean, eps_scale, eps0_scaled,
               n_epochs=15_000, lr=5e-4,
               lambda_data=1.0, lambda_hooke=0.05,
               lambda_bc=0.2,   lambda_smooth=0.005,
               print_every=1_000):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    history   = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss_data = nn.functional.mse_loss(model(X_train), y_train)
        res       = physics_residuals(
            model, X_phys, eps_mean, eps_scale, eps0_scaled
        )
        loss_phys = (lambda_hooke  * res["r_hooke"]
                   + lambda_bc     * res["r_bc"]
                   + lambda_smooth * res["r_smooth"])

        loss = lambda_data * loss_data + loss_phys
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append({
            "epoch":  epoch,
            "total":  loss.item(),
            "data":   loss_data.item(),
            "hooke":  res["r_hooke"].item(),
            "bc":     res["r_bc"].item(),
            "smooth": res["r_smooth"].item(),
        })

        if epoch % print_every == 0:
            print(f"Epoch {epoch:>6d} | total={loss.item():.4e} "
                  f"| data={loss_data.item():.4e} "
                  f"| hooke={res['r_hooke'].item():.4e} "
                  f"| bc={res['r_bc'].item():.4e}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 5. YOUNG'S MODULUS EXTRACTION
# NOTE: no @torch.no_grad() here — autograd must remain active
# ─────────────────────────────────────────────────────────────────────────────

def compute_youngs_modulus(model, cases, scaler_X):
    """
    For every unique (comp, SoC) case compute:
        E = dσ/dε  averaged over a small elastic strain window [0.001, 0.01]
    Chain rule: dσ/dε = (dσ / d(eps_scaled)) * (1 / eps_scale)
    """
    unique    = cases[["comp", "SoC", "x_Ni", "x_Mn", "x_Co"]].drop_duplicates()
    eps_scale = float(scaler_X.scale_[0])
    results   = []
    model.eval()                    # batchnorm / dropout off, but grad graph ON

    for _, row in unique.iterrows():
        N      = 20
        eps_np = np.linspace(0.001, 0.01, N, dtype=np.float32)

        x_raw = np.column_stack([
            eps_np,
            np.full(N, row["x_Ni"], dtype=np.float32),
            np.full(N, row["x_Mn"], dtype=np.float32),
            np.full(N, row["x_Co"], dtype=np.float32),
            np.full(N, row["SoC"],  dtype=np.float32),
        ])

        x_sc  = scaler_X.transform(x_raw).astype(np.float32)
        X_t   = torch.tensor(x_sc, requires_grad=True)   # grad ON
        sigma = model(X_t)                                # (N, 1)

        # dσ/d(eps_scaled) — gradient w.r.t. the scaled epsilon column
        g = torch.autograd.grad(sigma.sum(), X_t,
                                 create_graph=False,
                                 retain_graph=False)[0]   # (N, 5)

        dsigma_deps_scaled = g[:, 0].detach().numpy()
        E_GPa = float(dsigma_deps_scaled.mean()) / eps_scale

        results.append({
            "comp":  row["comp"],
            "SoC":   row["SoC"],
            "x_Ni":  round(row["x_Ni"], 4),
            "x_Mn":  round(row["x_Mn"], 4),
            "x_Co":  round(row["x_Co"], 4),
            "E_GPa": abs(E_GPa),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(history, save_path):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(df["epoch"], df["total"], label="Total")
    axes[0].semilogy(df["epoch"], df["data"],  label="Data")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss"); axes[0].legend()

    axes[1].semilogy(df["epoch"], df["hooke"],  label="Hooke")
    axes[1].semilogy(df["epoch"], df["bc"],     label="BC")
    axes[1].semilogy(df["epoch"], df["smooth"], label="Smooth")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Physics residuals"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_youngs_surface(E_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for comp, grp in E_df.groupby("comp"):
        grp_s = grp.sort_values("SoC")
        axes[0].plot(grp_s["SoC"], grp_s["E_GPa"], marker="o", label=f"NMC-{comp}")
    axes[0].set_xlabel("SoC"); axes[0].set_ylabel("Young's modulus (GPa)")
    axes[0].set_title("E vs SoC per composition"); axes[0].legend(fontsize=8)

    pivot = E_df.pivot_table(values="E_GPa", index="x_Ni", columns="SoC", aggfunc="mean")
    if pivot.shape[0] > 1 and pivot.shape[1] > 1:
        im = axes[1].imshow(pivot.values, aspect="auto", origin="lower",
                            extent=[pivot.columns.min(), pivot.columns.max(),
                                    pivot.index.min(),   pivot.index.max()])
        plt.colorbar(im, ax=axes[1], label="E (GPa)")
        axes[1].set_xlabel("SoC"); axes[1].set_ylabel("x_Ni")
        axes[1].set_title("E surface: Ni fraction x SoC")
    else:
        labels = [f"{r.comp}\nSoC={r.SoC}" for _, r in E_df.iterrows()]
        axes[1].bar(range(len(E_df)), E_df["E_GPa"].values)
        axes[1].set_xticks(range(len(E_df)))
        axes[1].set_xticklabels(labels, fontsize=7)
        axes[1].set_ylabel("E (GPa)"); axes[1].set_title("E per simulation case")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_stress_strain(model, scaler_X, df, n_cases=None,
                       save_path="pinn_stress_strain.png"):
    cases = df[["comp", "SoC"]].drop_duplicates()
    if n_cases is not None:
        cases = cases.head(n_cases)

    nplots = len(cases)
    ncols = min(4, nplots)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    model.eval()
    for ax, (_, case) in zip(axes_flat, cases.iterrows()):
        sub    = df[(df["comp"] == case["comp"]) & (df["SoC"] == case["SoC"])]
        eps_md = sub["strain"].values
        sig_md = sub["stress_xx"].values
        sig_0  = float(sub["stress_initial"].iloc[0])
        eps_g  = np.linspace(eps_md.min(), eps_md.max(), 200, dtype=np.float32)

        x_raw = np.column_stack([
            eps_g,
            np.full_like(eps_g, sub["x_Ni"].iloc[0]),
            np.full_like(eps_g, sub["x_Mn"].iloc[0]),
            np.full_like(eps_g, sub["x_Co"].iloc[0]),
            np.full_like(eps_g, case["SoC"]),
        ])
        x_sc = scaler_X.transform(x_raw).astype(np.float32)
        with torch.no_grad():
            # The model predicts incremental stress; restore the MD offset for
            # an absolute-stress comparison in this diagnostic.
            sig_p = model(torch.tensor(x_sc)).numpy().flatten() + sig_0

        ax.scatter(eps_md, sig_md, s=8, alpha=0.5, label="MD",   color="steelblue")
        ax.plot(eps_g,     sig_p,  lw=1.8,          label="PINN", color="coral")
        ax.set_title(f"NMC-{case['comp']}, SoC={case['SoC']}")
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress_xx (GPa)")
        ax.legend(fontsize=8)

    # Hide empty cells when the number of cases does not fill the final row.
    for ax in axes_flat[nplots:]:
        ax.set_visible(False)

    fig.suptitle("PINN vs MD stress-strain -- all parsed cases", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load & filter to elastic regime
    df    = build_dataset(DATA_DIR)
    # Work with incremental stress so delta_sigma(epsilon=0)=0 remains valid
    # for simulations that contain a nonzero initial/prestress value.
    df["stress_initial"] = df.groupby("filename")["stress_xx"].transform("first")
    df["delta_stress"] = df["stress_xx"] - df["stress_initial"]
    df_el = df[df["strain"].abs() < ELASTIC_STRAIN_LIMIT].copy()
    print(f"Elastic-regime rows (|strain| < {ELASTIC_STRAIN_LIMIT}): {len(df_el)}")

    feat_cols  = ["strain", "x_Ni", "x_Mn", "x_Co", "SoC"]
    target_col = "delta_stress"

    X = df_el[feat_cols].values.astype(np.float32)
    y = df_el[target_col].values.astype(np.float32).reshape(-1, 1)

    scaler_X = StandardScaler()
    X_sc     = scaler_X.fit_transform(X)
    eps_mean = float(scaler_X.mean_[0])
    eps_scale = float(scaler_X.scale_[0])
    eps0_scaled = (0.0 - eps_mean) / eps_scale

    X_tr, X_va, y_tr, y_va = train_test_split(X_sc, y, test_size=0.15, random_state=42)

    X_train = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_val   = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_val   = torch.tensor(y_va, dtype=torch.float32).to(device)

    n_phys = min(len(X_train) * 3, 10_000)
    idx    = np.random.choice(len(X_tr), n_phys, replace=True)
    X_phys = torch.tensor(X_tr[idx], dtype=torch.float32).to(device)

    # Build & train
    model = PINN(n_hidden=N_HIDDEN, n_neurons=N_NEURONS).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    history = train_pinn(
        model, X_train, y_train, X_phys,
        eps_mean=eps_mean, eps_scale=eps_scale,
        eps0_scaled=eps0_scaled,
        n_epochs=N_EPOCHS, lr=LR,
        lambda_data=LAMBDA_DATA, lambda_hooke=LAMBDA_HOOKE,
        lambda_bc=LAMBDA_BC,     lambda_smooth=LAMBDA_SMOOTH,
        print_every=PRINT_EVERY,
    )

    # Validation
    model.eval()
    with torch.no_grad():
        val_mse = nn.functional.mse_loss(model(X_val), y_val).item()
    print(f"\nValidation MSE: {val_mse:.4e} GPa^2")

    # Move to CPU before autograd extraction (works on both CPU and GPU setups)
    model.to("cpu")

    # Extract Young's modulus (grad graph must be active — no torch.no_grad)
    print("\nExtracting Young's modulus via autograd ...")
    E_df = compute_youngs_modulus(model, df_el, scaler_X)

    print("\nYoung's modulus per case:")
    print(E_df.to_string(index=False))

    csv_path = OUTPUT_DIR / "nmc_E_pinn.csv"
    E_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save model checkpoint
    ckpt_path = OUTPUT_DIR / "nmc_pinn_model.pt"
    torch.save({
        "model_state":  model.state_dict(),
        "scaler_mean":  scaler_X.mean_,
        "scaler_scale": scaler_X.scale_,
        "feat_cols":    feat_cols,
        "target_col":   target_col,
        "stress_output": "increment_from_initial",
        "n_hidden":     N_HIDDEN,
        "n_neurons":    N_NEURONS,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    # Plots
    plot_loss(history,        str(OUTPUT_DIR / "pinn_loss.png"))
    plot_youngs_surface(E_df, str(OUTPUT_DIR / "pinn_E_surface.png"))
    plot_stress_strain(model, scaler_X, df_el,
                       save_path=str(OUTPUT_DIR / "pinn_stress_strain.png"))

    print("\nAll done.")


if __name__ == "__main__":
    main()
