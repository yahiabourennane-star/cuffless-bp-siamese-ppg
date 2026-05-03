import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import compute_sbp_tail_multipliers
from evaluate import (
    apply_linear_calibration,
    apply_sbp_residual_corrector,
    build_sbp_residual_feature_matrix,
    fit_linear_calibration,
    fit_sbp_residual_corrector,
)
from metrics import compute_metrics
from run_ensemble import load_ppg_eval_data, run_multi_anchor_for_model


def parse_csv_list(raw, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def bhs_percentages(abs_err):
    abs_err = np.asarray(abs_err, dtype=np.float32)
    return {
        "le5": float((abs_err <= 5.0).mean() * 100.0),
        "le10": float((abs_err <= 10.0).mean() * 100.0),
        "le15": float((abs_err <= 15.0).mean() * 100.0),
    }


def summarize(pred, tgt):
    pred = np.asarray(pred, dtype=np.float32)
    tgt = np.asarray(tgt, dtype=np.float32)
    m = compute_metrics(pred, tgt)
    sbp_abs = np.abs(pred[:, 0] - tgt[:, 0])
    dbp_abs = np.abs(pred[:, 1] - tgt[:, 1])
    return {
        "mae_sbp": float(m["mae_sbp"]),
        "mae_dbp": float(m["mae_dbp"]),
        "combined_mae": float(m["combined_mae"]),
        "rmse_sbp": float(m["rmse_sbp"]),
        "rmse_dbp": float(m["rmse_dbp"]),
        "me_sbp": float(m["me_sbp"]),
        "me_dbp": float(m["me_dbp"]),
        "std_sbp": float(m["std_sbp"]),
        "std_dbp": float(m["std_dbp"]),
        "r_sbp": float(m["r_sbp"]),
        "r_dbp": float(m["r_dbp"]),
        "bhs_sbp": str(m["bhs_sbp"]),
        "bhs_dbp": str(m["bhs_dbp"]),
        "sbp_bhs": bhs_percentages(sbp_abs),
        "dbp_bhs": bhs_percentages(dbp_abs),
    }


def score_key(metrics_dict):
    sbp_bhs = metrics_dict["sbp_bhs"]
    hit_grade_b = (
        sbp_bhs["le5"] >= 50.0
        and sbp_bhs["le10"] >= 75.0
        and sbp_bhs["le15"] >= 90.0
    )
    return (
        int(hit_grade_b),
        sbp_bhs["le5"],
        sbp_bhs["le10"],
        sbp_bhs["le15"],
        -metrics_dict["mae_sbp"],
        -metrics_dict["combined_mae"],
        metrics_dict["r_sbp"],
    )


def format_summary(label, metrics_dict):
    sbp = metrics_dict["sbp_bhs"]
    return (
        f"{label}: SBP={metrics_dict['mae_sbp']:.2f}  DBP={metrics_dict['mae_dbp']:.2f}  "
        f"Comb={metrics_dict['combined_mae']:.2f}  "
        f"SBP<=5={sbp['le5']:.2f}%  <=10={sbp['le10']:.2f}%  <=15={sbp['le15']:.2f}%  "
        f"BHS={metrics_dict['bhs_sbp']}/{metrics_dict['bhs_dbp']}  r={metrics_dict['r_sbp']:.3f}/{metrics_dict['r_dbp']:.3f}"
    )


def discover_members(ensemble_dir):
    members = sorted(ensemble_dir.glob("seed_*/best_model.pt"))
    if len(members) < 2:
        raise RuntimeError(f"Need at least two member checkpoints under {ensemble_dir}")
    return members


def cache_paths(ensemble_dir, tta):
    suffix = f"tta{tta}" if tta > 0 else "notta"
    return {
        "val_pred": ensemble_dir / f"cache_val_pred_{suffix}.npy",
        "val_tgt": ensemble_dir / "cache_val_tgt.npy",
        "test_pred": ensemble_dir / f"cache_test_pred_{suffix}.npy",
        "test_tgt": ensemble_dir / "cache_test_tgt.npy",
        "meta": ensemble_dir / f"cache_meta_{suffix}.json",
    }


def compute_or_load_ensemble_predictions(ensemble_dir, data_dir, eval_batch_size, tta, force_recompute):
    paths = cache_paths(ensemble_dir, tta)
    if not force_recompute and all(p.exists() for p in paths.values()):
        print("Loading cached ensemble val/test predictions...")
        return (
            np.load(paths["val_pred"]).astype(np.float32),
            np.load(paths["val_tgt"]).astype(np.float32),
            np.load(paths["test_pred"]).astype(np.float32),
            np.load(paths["test_tgt"]).astype(np.float32),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    members = discover_members(ensemble_dir)
    print(f"Recomputing ensemble predictions on {device} from {len(members)} members...")

    _, val_ds, test_ds, _, _, _ = load_ppg_eval_data(str(data_dir))
    hf_dim = test_ds.hand_feature_dim
    n_ch = test_ds.num_channels

    val_preds = []
    test_preds = []
    val_tgt = None
    test_tgt = None

    for ckpt in members:
        print(f"  member: {ckpt.parent.name}")
        vp, vt = run_multi_anchor_for_model(
            str(ckpt), val_ds, device, eval_batch_size, tta, hf_dim, n_ch
        )
        tp, tt = run_multi_anchor_for_model(
            str(ckpt), test_ds, device, eval_batch_size, tta, hf_dim, n_ch
        )
        val_preds.append(vp.astype(np.float32))
        test_preds.append(tp.astype(np.float32))
        if val_tgt is None:
            val_tgt = vt.astype(np.float32)
        if test_tgt is None:
            test_tgt = tt.astype(np.float32)

    val_ensemble = np.mean(np.stack(val_preds, axis=0), axis=0).astype(np.float32)
    test_ensemble = np.mean(np.stack(test_preds, axis=0), axis=0).astype(np.float32)

    np.save(paths["val_pred"], val_ensemble)
    np.save(paths["val_tgt"], val_tgt)
    np.save(paths["test_pred"], test_ensemble)
    np.save(paths["test_tgt"], test_tgt)
    paths["meta"].write_text(
        json.dumps(
            {
                "members": [str(p.resolve()) for p in members],
                "tta": int(tta),
                "eval_batch_size": int(eval_batch_size),
                "data_dir": str(Path(data_dir).resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return val_ensemble, val_tgt, test_ensemble, test_tgt


def main():
    parser = argparse.ArgumentParser(
        description="Validation-driven sweep of the SBP residual corrector for the finished v11 ensemble."
    )
    parser.add_argument("--ensemble_dir", default=r".\ensemble_v11_abpbeat")
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v11")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--tta", type=int, default=10)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--out_json", default=r".\ensemble_v11_abpbeat\sweep_bhs_sbp_residual.json")
    parser.add_argument("--out_txt", default=r".\ensemble_v11_abpbeat\sweep_bhs_sbp_residual.txt")
    parser.add_argument("--alphas", default="32,64,96,128")
    parser.add_argument("--clips", default="10,12,14")
    parser.add_argument("--model_types", default="ridge,piecewise_ridge")
    parser.add_argument("--piecewise_edges_list", default="108,128;110,130;112,132")
    parser.add_argument("--piecewise_min_samples", default="512,1024")
    parser.add_argument("--piecewise_blend_widths", default="4,6")
    parser.add_argument(
        "--tail_configs",
        default="off;90,110,130,150,1.15,1.60;95,115,125,145,1.20,1.80",
        help="Semicolon-separated configs: off or tail_low,shoulder_low,shoulder_high,tail_high,shoulder_boost,tail_boost",
    )
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    data_dir = Path(args.data_dir)
    out_json = Path(args.out_json)
    out_txt = Path(args.out_txt)

    t0 = time.time()
    alphas = parse_csv_list(args.alphas, float)
    clips = parse_csv_list(args.clips, float)
    model_types = [x.strip() for x in args.model_types.split(",") if x.strip()]
    edge_options = [x.strip() for x in args.piecewise_edges_list.split(";") if x.strip()]
    min_sample_options = parse_csv_list(args.piecewise_min_samples, int)
    blend_options = parse_csv_list(args.piecewise_blend_widths, float)

    tail_configs = []
    for raw in [x.strip() for x in args.tail_configs.split(";") if x.strip()]:
        if raw.lower() == "off":
            tail_configs.append(
                {
                    "name": "off",
                    "enabled": False,
                    "tail_low": 90.0,
                    "shoulder_low": 110.0,
                    "shoulder_high": 130.0,
                    "tail_high": 150.0,
                    "shoulder_boost": 1.0,
                    "tail_boost": 1.0,
                }
            )
            continue
        vals = [float(v.strip()) for v in raw.split(",")]
        if len(vals) != 6:
            raise ValueError(f"Bad tail config: {raw}")
        tail_low, shoulder_low, shoulder_high, tail_high, shoulder_boost, tail_boost = vals
        tail_configs.append(
            {
                "name": raw,
                "enabled": True,
                "tail_low": tail_low,
                "shoulder_low": shoulder_low,
                "shoulder_high": shoulder_high,
                "tail_high": tail_high,
                "shoulder_boost": shoulder_boost,
                "tail_boost": tail_boost,
            }
        )

    _, val_ds, test_ds, _, _, _ = load_ppg_eval_data(str(data_dir))
    val_pred, val_tgt, test_pred, test_tgt = compute_or_load_ensemble_predictions(
        ensemble_dir, data_dir, args.eval_batch_size, args.tta, args.force_recompute
    )

    cal_scale, cal_bias = fit_linear_calibration(val_pred, val_tgt, targets="sbp")
    val_cal = apply_linear_calibration(val_pred, cal_scale, cal_bias)
    test_cal = apply_linear_calibration(test_pred, cal_scale, cal_bias)
    val_feat = build_sbp_residual_feature_matrix(val_ds, val_cal)
    test_feat = build_sbp_residual_feature_matrix(test_ds, test_cal)

    baseline = {
        "ensemble_tta": summarize(test_pred, test_tgt),
        "ensemble_tta_cal": summarize(test_cal, test_tgt),
    }

    results = []
    combo_count = 0
    for model_type in model_types:
        for alpha, clip, tail_cfg in itertools.product(alphas, clips, tail_configs):
            weights = None
            if tail_cfg["enabled"]:
                weights = compute_sbp_tail_multipliers(
                    val_tgt[:, 0],
                    shoulder_low=tail_cfg["shoulder_low"],
                    tail_low=tail_cfg["tail_low"],
                    shoulder_high=tail_cfg["shoulder_high"],
                    tail_high=tail_cfg["tail_high"],
                    shoulder_boost=tail_cfg["shoulder_boost"],
                    tail_boost=tail_cfg["tail_boost"],
                ).astype(np.float32)

            if model_type == "ridge":
                combo_count += 1
                params = {
                    "model_type": model_type,
                    "alpha": float(alpha),
                    "clip_mmHg": float(clip),
                    "tail_weighting": bool(tail_cfg["enabled"]),
                    "tail_config": tail_cfg,
                }
                model_state = fit_sbp_residual_corrector(
                    val_feat,
                    val_cal[:, 0],
                    val_tgt[:, 0],
                    alpha=alpha,
                    sample_weight=weights,
                    clip_mmHg=clip,
                    model_type="ridge",
                )
                val_out = apply_sbp_residual_corrector(val_cal, val_feat, model_state)
                test_out = apply_sbp_residual_corrector(test_cal, test_feat, model_state)
                val_metrics = summarize(val_out, val_tgt)
                test_metrics = summarize(test_out, test_tgt)
                results.append(
                    {
                        "params": params,
                        "val": val_metrics,
                        "test": test_metrics,
                        "score": score_key(val_metrics),
                    }
                )
                continue

            for edges, min_samples, blend in itertools.product(
                edge_options, min_sample_options, blend_options
            ):
                combo_count += 1
                params = {
                    "model_type": model_type,
                    "alpha": float(alpha),
                    "clip_mmHg": float(clip),
                    "piecewise_edges": edges,
                    "piecewise_min_samples": int(min_samples),
                    "piecewise_blend_width": float(blend),
                    "tail_weighting": bool(tail_cfg["enabled"]),
                    "tail_config": tail_cfg,
                }
                model_state = fit_sbp_residual_corrector(
                    val_feat,
                    val_cal[:, 0],
                    val_tgt[:, 0],
                    alpha=alpha,
                    sample_weight=weights,
                    clip_mmHg=clip,
                    model_type="piecewise_ridge",
                    piecewise_edges=edges,
                    piecewise_min_samples=min_samples,
                    piecewise_blend_width=blend,
                )
                val_out = apply_sbp_residual_corrector(val_cal, val_feat, model_state)
                test_out = apply_sbp_residual_corrector(test_cal, test_feat, model_state)
                val_metrics = summarize(val_out, val_tgt)
                test_metrics = summarize(test_out, test_tgt)
                results.append(
                    {
                        "params": params,
                        "val": val_metrics,
                        "test": test_metrics,
                        "score": score_key(val_metrics),
                    }
                )

    results.sort(key=lambda row: row["score"], reverse=True)
    best = results[0]

    default_params = {
        "model_type": "piecewise_ridge",
        "alpha": 64.0,
        "clip_mmHg": 12.0,
        "piecewise_edges": "110,130",
        "piecewise_min_samples": 1024,
        "piecewise_blend_width": 6.0,
        "tail_weighting": True,
        "tail_config": {
            "name": "90,110,130,150,1.15,1.60",
            "enabled": True,
            "tail_low": 90.0,
            "shoulder_low": 110.0,
            "shoulder_high": 130.0,
            "tail_high": 150.0,
            "shoulder_boost": 1.15,
            "tail_boost": 1.60,
        },
    }

    default_match = None
    for row in results:
        if row["params"] == default_params:
            default_match = row
            break

    payload = {
        "ensemble_dir": str(ensemble_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "tta": int(args.tta),
        "eval_batch_size": int(args.eval_batch_size),
        "combo_count": int(combo_count),
        "elapsed_sec": float(time.time() - t0),
        "baseline": baseline,
        "default_config": default_match,
        "best_by_validation": best,
        "top10_by_validation": results[:10],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"Sweep combos: {combo_count}",
        f"Elapsed: {payload['elapsed_sec']:.1f}s",
        "",
        format_summary("Baseline Ensemble + TTA", baseline["ensemble_tta"]),
        format_summary("Baseline Ensemble + TTA + Cal", baseline["ensemble_tta_cal"]),
    ]
    if default_match is not None:
        lines.extend(
            [
                "",
                "Default residual config:",
                json.dumps(default_match["params"], indent=2),
                format_summary("  Val", default_match["val"]),
                format_summary("  Test", default_match["test"]),
            ]
        )
    lines.extend(
        [
            "",
            "Best by validation objective:",
            json.dumps(best["params"], indent=2),
            format_summary("  Val", best["val"]),
            format_summary("  Test", best["test"]),
            "",
            "Top 10 by validation objective:",
        ]
    )
    for i, row in enumerate(results[:10], start=1):
        lines.append(f"{i:02d}. {json.dumps(row['params'])}")
        lines.append(f"    {format_summary('Val', row['val'])}")
        lines.append(f"    {format_summary('Test', row['test'])}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved JSON summary to {out_json}")
    print(f"Saved text summary to {out_txt}")
    print("")
    print(format_summary("Best test (selected by validation)", best["test"]))


if __name__ == "__main__":
    main()
