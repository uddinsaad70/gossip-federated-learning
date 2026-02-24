"""
test_step5.py — Phase 5: Quality Assessment & Byzantine Detection
PDF Section 3.5 — প্রতিটা calculation বিস্তারিত দেখায়
Run with: python test_step5.py
"""
import torch, copy, numpy as np
from topology import create_devices, build_topology
from data_loader import load_mnist, distribute_iid
from grad_compression import compress_gradient
from privacy import apply_differential_privacy
from gossip import gossip_exchange
from byzantine import assess_received_gradients

MODEL_SIZE      = 421_642
SHOW_DETAIL_FOR = [1]       # কোন device এর full calculation দেখাবে
SHOW_N_ELEMENTS = 5         # vector এ কতটা element print করবে

def sep(c="="): print(f"\n{c*65}")
def section(t): sep(); print(f"  {t}"); sep()
def sub(t): sep("─"); print(f"  {t}"); sep("─")

def fv(a, n=SHOW_N_ELEMENTS):
    v = [f"{x:+.5f}" for x in a[:n]]
    return "[" + ", ".join(v) + (", ..." if len(a)>n else "") + "]"

def flatten_comp(d): return np.concatenate([p["data"].detach().numpy().flatten() for p in d.values()])
def flatten_msg(m):
    if m.get("compressed"):
        return np.concatenate([t.detach().numpy().flatten() for t in m["compressed"].values()])
    return np.concatenate([t.detach().numpy().flatten() for t in m["gradient"].values()])

# ── Test 1 ────────────────────────────────────
def calc_cosine(A, B, la="A", lb="B"):
    dot   = float(np.dot(A,B))
    na    = float(np.linalg.norm(A))
    nb    = float(np.linalg.norm(B))
    denom = na*nb
    cs    = dot/denom if denom>0 else 0.0

    print(f"\n    Test 1: Cosine Similarity")
    print(f"      formula: cos_sim({la},{lb}) = ({la}·{lb}) / (‖{la}‖ × ‖{lb}‖)")
    terms = " + ".join(f"({A[i]:+.5f})×({B[i]:+.5f})" for i in range(min(3,len(A))))
    pvals = " + ".join(f"{A[i]*B[i]:+.8f}" for i in range(min(3,len(A))))
    print(f"      dot product  = {terms} + ···")
    print(f"                   = {pvals} + ···")
    print(f"                   = {dot:.8f}")
    print(f"      ‖{la}‖ = {na:.8f}")
    print(f"      ‖{lb}‖ = {nb:.8f}")
    print(f"      cos_sim = {dot:.8f} / ({na:.8f} × {nb:.8f})")
    print(f"              = {dot:.8f} / {denom:.8f}")
    print(f"              = {cs:.6f}")

    if cs < -0.5:
        msg = f"⚠  {cs:.4f} < -0.5 → DIRECTIONAL ATTACK → quality = 0"
    elif cs > 0.7: msg = f"✓  {cs:.4f} → Good similarity"
    elif cs > 0.3: msg = f"~  {cs:.4f} → Moderate similarity"
    else:          msg = f"~  {cs:.4f} → Low (noise-dominated)"
    print(f"      Interpretation: {msg}")
    return cs, na, nb

# ── Test 2 ────────────────────────────────────
def calc_magnitude(na, nb, la="A", lb="B"):
    mn  = min(na, nb)
    mx  = max(na, nb)
    rat = mn/mx if mx>0 else 1.0
    print(f"\n    Test 2: Magnitude Ratio")
    print(f"      formula: mag_ratio = min(‖{la}‖,‖{lb}‖) / max(‖{la}‖,‖{lb}‖)")
    print(f"             = min({na:.8f}, {nb:.8f})")
    print(f"               ─────────────────────────────────────────────")
    print(f"               max({na:.8f}, {nb:.8f})")
    print(f"             = {mn:.8f} / {mx:.8f}")
    print(f"             = {rat:.6f}")
    if rat < 0.1:   msg = f"⚠  {rat:.4f} < 0.1 → SCALING ATTACK → quality = 0"
    elif rat > 0.5: msg = f"✓  {rat:.4f} > 0.5 → Acceptable"
    else:           msg = f"~  {rat:.4f} → Low ratio"
    print(f"      Interpretation: {msg}")
    return rat

# ── Test 3 ────────────────────────────────────
def calc_outlier(A, B):
    diff = np.abs(B-A)
    Q1   = float(np.percentile(diff,25))
    Q3   = float(np.percentile(diff,75))
    IQR  = Q3-Q1
    thr  = Q3 + 1.5*IQR
    outs = int(np.sum(diff>thr))
    pct  = outs/len(diff)

    print(f"\n    Test 3: Outlier Detection (IQR)")
    print(f"      |B - A|  = element-wise absolute differences")
    print(f"               = {fv(diff)}")
    print(f"      Q1 (25th pct) = {Q1:.8f}")
    print(f"      Q3 (75th pct) = {Q3:.8f}")
    print(f"      IQR = Q3 - Q1 = {IQR:.8f}")
    print(f"      Threshold = Q3 + 1.5 × IQR")
    print(f"                = {Q3:.8f} + 1.5 × {IQR:.8f}")
    print(f"                = {Q3:.8f} + {1.5*IQR:.8f}")
    print(f"                = {thr:.8f}")
    print(f"      Outliers (diff > {thr:.8f}): {outs} out of {len(diff)}")
    print(f"      Outlier pct = {outs}/{len(diff)} = {pct*100:.4f}%")
    if pct>0.30: msg = f"⚠  {pct*100:.2f}% > 30% → ANOMALY → quality = 0"
    else:        msg = f"✓  {pct*100:.2f}% ≤ 30% → Normal"
    print(f"      Interpretation: {msg}")
    return pct

# ── Quality Score ─────────────────────────────
def calc_quality(cs, mr, op, sid):
    print(f"\n    Quality Score Logic (PDF Section 3.5.1):")
    print(f"      if outlier_pct > 30%   : {op*100:.4f}% {'> 30%? YES' if op>0.30 else '≤ 30%  ✓'}")
    print(f"      if cos_sim    < -0.5   : {cs:.4f}   {'< -0.5? YES' if cs<-0.5 else '≥ -0.5 ✓'}")
    print(f"      if mag_ratio  < 0.1    : {mr:.4f}   {'< 0.1? YES' if mr<0.1 else '≥ 0.1  ✓'}")

    if op>0.30 or cs<-0.5 or mr<0.1:
        q = 0.0
        reason = ("outlier_pct>30%" if op>0.30 else
                  "cos_sim<-0.5" if cs<-0.5 else "mag_ratio<0.1")
        print(f"\n      Condition met ({reason}) → Quality = 0.0  ← BYZANTINE FLAG")
    else:
        q = 0.6*cs + 0.4*mr
        print(f"\n      All conditions OK → compute score:")
        print(f"      Quality = 0.6 × cos_sim + 0.4 × mag_ratio")
        print(f"              = 0.6 × {cs:.6f} + 0.4 × {mr:.6f}")
        print(f"              = {0.6*cs:.6f} + {0.4*mr:.6f}")
        print(f"              = {q:.6f}")
        rating = "Excellent" if q>0.75 else ("Good" if q>0.50 else "Acceptable")
        print(f"      Rating: {rating}")
    return q

# ── Reputation Update ─────────────────────────
def calc_reputation(rep_old, q, sid):
    rep_new = round(0.8*rep_old + 0.2*q, 4)
    print(f"\n    Reputation Update (PDF Section 3.5.4):")
    print(f"      rep_new = 0.8 × rep_old + 0.2 × Quality")
    print(f"              = 0.8 × {rep_old:.4f} + 0.2 × {q:.6f}")
    print(f"              = {0.8*rep_old:.6f} + {0.2*q:.6f}")
    print(f"              = {rep_new:.4f}")
    return rep_new

# ── Full evaluation for one device ────────────
def evaluate_device_detailed(device, msgs, own_comp, all_devs):
    byz_ids = {d.id for d in all_devs if d.is_byzantine}
    own_flat = flatten_comp(own_comp)

    print(f"\n  Own gradient (Device {device.id}):")
    print(f"    Domain : compressed / pre-noise (Phase 2 output)")
    print(f"    ‖own‖  = {np.linalg.norm(own_flat):.8f}")
    print(f"    Values : {fv(own_flat)}")

    results = {}
    for msg in sorted(msgs, key=lambda m: m["sender"]):
        sid    = msg["sender"]
        is_byz = sid in byz_ids
        byz_tag= "  *** BYZANTINE DEVICE ***" if is_byz else ""

        print(f"\n  {'━'*63}")
        print(f"  Device {device.id} evaluating Device {sid}{byz_tag}")
        print(f"  {'━'*63}")

        recv_flat = flatten_msg(msg)
        n  = min(len(own_flat), len(recv_flat))
        A, B = own_flat[:n], recv_flat[:n]

        domain = "compressed/pre-noise" if msg.get("compressed") else "noisy/fallback"
        print(f"\n    Input Vectors ({domain}):")
        print(f"    Own  (D{device.id:2d}): ‖A‖={np.linalg.norm(A):.8f}  {fv(A)}")
        print(f"    Recv (D{sid:2d}): ‖B‖={np.linalg.norm(B):.8f}  {fv(B)}")

        cs, na, nb = calc_cosine(A, B, f"D{device.id}", f"D{sid}")
        mr         = calc_magnitude(na, nb, f"D{device.id}", f"D{sid}")
        op         = calc_outlier(A, B)
        q          = calc_quality(cs, mr, op, sid)
        rep_old    = device.reputation.get(sid, 1.0)
        rep_new    = calc_reputation(rep_old, q, sid)
        device.reputation[sid] = rep_new

        flag = "BYZANTINE" if q==0.0 else "trusted"
        box  = "⚠  BYZANTINE DETECTED" if q==0.0 else "✓  trusted"
        print(f"\n    ┌──────────────────────────────────────────────┐")
        print(f"    │  Result: Device {sid:2d} → {box:<24} │")
        print(f"    │  cos_sim={cs:+.4f}  mag_ratio={mr:.4f}  quality={q:.4f}  │")
        print(f"    │  reputation: {rep_old:.4f} → {rep_new:.4f}                    │")
        print(f"    └──────────────────────────────────────────────┘")

        results[sid] = {"cos_sim":cs,"mag_ratio":mr,"outlier_pct":op,
                        "quality":q,"rep_old":rep_old,"rep_new":rep_new,
                        "is_byzantine":q==0.0}
    return results


def main():
    section("PHASE 0 — Initialization")
    devices    = create_devices()
    G, manager = build_topology(devices)
    W0 = copy.deepcopy(devices[0].model.state_dict())
    for d in devices:
        d.model.load_state_dict(W0)
        d.init_reputation()
    X_train, y_train, X_test, y_test = load_mnist()
    distribute_iid(X_train, y_train, devices)
    byz_d = next(d for d in devices if d.is_byzantine)
    print(f"  {len(devices)} devices  |  Byzantine: Device {byz_d.id}")
    print(f"  Full calculation shown for: Device {SHOW_DETAIL_FOR}")

    section("PHASE 1→3 — Training, Compression, DP")
    noisy_grad, comp_grad = {}, {}
    for d in devices:
        raw          = d.local_train(batch_size=64)
        comp         = compress_gradient(raw, d, MODEL_SIZE)
        noisy, _     = apply_differential_privacy(comp)
        noisy_grad[d.id] = noisy
        comp_grad[d.id]  = comp
    print(f"  Done.")

    section("PHASE 4 — Gossip Exchange")
    received = gossip_exchange(devices, noisy_grad, 1, comp_grad)
    print(f"  Done.")

    section("PHASE 5 — Quality Assessment & Byzantine Detection (PDF 3.5)")
    print(f"  Detection domain: compressed/pre-noise (Phase 2 output)")
    print(f"  Reason: noise_L2={0.1*(MODEL_SIZE**0.5):.1f} >> signal_L2≈0.03")
    print(f"  Sign-flip in noisy domain: cos_sim≈0 (undetectable)")
    print(f"  Sign-flip in compressed  : cos_sim≈-1 (detected ✓)")

    all_quality = {}
    for d in devices:
        msgs  = received.get(d.id, [])
        own_n = noisy_grad.get(d.id)
        own_c = comp_grad.get(d.id)
        if d.id in SHOW_DETAIL_FOR:
            sub(f"DEVICE {d.id} — Full PDF-Style Calculation (Section 3.5.2)")
            for nid in d.neighbors: d.reputation[nid] = 1.0
            all_quality[d.id] = evaluate_device_detailed(d, msgs, own_c, devices)
        else:
            all_quality[d.id] = assess_received_gradients(d, msgs, own_n, own_c)

    # Summary
    section("BYZANTINE DETECTION SUMMARY")
    print(f"\n  {'Device':<22} {'Evaluated':>10} {'Byz Detected':>25}")
    print(f"  {'-'*60}")
    total = 0
    for d in devices:
        scores  = all_quality[d.id]
        byz_lst = [sid for sid,i in scores.items() if i["is_byzantine"]]
        total  += len(byz_lst)
        tag     = " [BYZ]" if d.is_byzantine else ""
        print(f"  Device {d.id:2d}{tag:7s}        {len(scores):>10}  {str(byz_lst):>25}")
    print(f"\n  Total Byzantine flags: {total}")

    section(f"HONEST DEVICES RATING Device {byz_d.id} [BYZANTINE]")
    print(f"\n  {'Evaluator':<15} {'cos_sim':>9} {'mag_ratio':>10} {'quality':>9} {'rep_new':>9} {'flag':>12}")
    print(f"  {'-'*58}")
    for d in devices:
        if d.is_byzantine or byz_d.id not in all_quality[d.id]: continue
        i = all_quality[d.id][byz_d.id]
        flag = "BYZANTINE" if i["is_byzantine"] else "trusted"
        print(f"  Device {d.id:2d}           {i['cos_sim']:>9.4f} {i['mag_ratio']:>10.4f} "
              f"{i['quality']:>9.4f} {i['rep_new']:>9.4f} {flag:>12}")

    section("REPUTATION AFTER ROUND 1")
    print(f"\n  {'Device':<22} {'Min rep':>10} {'Max rep':>10} {'Avg rep':>10}")
    print(f"  {'-'*54}")
    for d in devices:
        if not d.reputation: continue
        reps = list(d.reputation.values())
        tag  = " [BYZ]" if d.is_byzantine else ""
        print(f"  Device {d.id:2d}{tag:7s}      "
              f"{min(reps):>10.4f} {max(reps):>10.4f} {sum(reps)/len(reps):>10.4f}")

    print(f"\n[OK] Phase 5 complete → Phase 6 (Aggregation)")

if __name__ == "__main__":
    main()