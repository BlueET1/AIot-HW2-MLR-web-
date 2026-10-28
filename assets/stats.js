/*
 * stats.js — 多元線性回歸所需的數值與統計工具（純 JavaScript，無外部相依）
 * 提供：資料切分、矩陣反矩陣、t / F / χ² 分佈、OLS（含信賴/預測區間）、
 *       VIF、SelectKBest、RFE、LassoCV、迴歸診斷檢定。
 */
(function (root) {
  'use strict';

  // ---------- 亂數與資料切分 ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function trainTestSplit(n, testSize, seed) {
    const idx = Array.from({ length: n }, (_, i) => i);
    const rand = mulberry32(seed);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const nTest = Math.ceil(n * testSize);
    return { test: idx.slice(0, nTest), train: idx.slice(nTest) };
  }

  // ---------- 基本統計 ----------
  function mean(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s / a.length;
  }

  function std(a, ddof = 0) {
    const m = mean(a);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
    return Math.sqrt(s / (a.length - ddof));
  }

  function median(a) {
    const s = Float64Array.from(a).sort();
    const h = s.length >> 1;
    return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
  }

  function quantile(a, q) {
    const s = Float64Array.from(a).sort();
    const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  }

  function pearson(a, b) {
    const ma = mean(a), mb = mean(b);
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < a.length; i++) {
      const da = a[i] - ma, db = b[i] - mb;
      sab += da * db; saa += da * da; sbb += db * db;
    }
    return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
  }

  // ---------- 線性代數 ----------
  // Gauss-Jordan（部分主元）；奇異時回傳 null
  function invert(A) {
    const n = A.length;
    let scale = 0;
    for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(A[i][i]));
    const tol = 1e-11 * (scale || 1);
    const M = A.map((row, i) => {
      const r = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) r[j] = row[j];
      r[n + i] = 1;
      return r;
    });
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (Math.abs(M[piv][c]) < tol) return null;
      [M[c], M[piv]] = [M[piv], M[c]];
      const d = M[c][c];
      for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c];
        if (f !== 0) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map(row => Array.from(row.subarray(n)));
  }

  function matVec(A, v) {
    return A.map(row => {
      let s = 0;
      for (let j = 0; j < v.length; j++) s += row[j] * v[j];
      return s;
    });
  }

  function quadForm(A, v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) {
      let r = 0;
      for (let j = 0; j < v.length; j++) r += A[i][j] * v[j];
      s += v[i] * r;
    }
    return s;
  }

  // Z'Z 與 Z'y（Z 為列向量陣列）
  function crossProducts(Z, y) {
    const k = Z[0].length;
    const G = Array.from({ length: k }, () => new Float64Array(k));
    const g = new Float64Array(k);
    for (let i = 0; i < Z.length; i++) {
      const z = Z[i];
      for (let a = 0; a < k; a++) {
        const za = z[a];
        if (za === 0) continue;
        g[a] += za * y[i];
        const Ga = G[a];
        for (let b = a; b < k; b++) Ga[b] += za * z[b];
      }
    }
    for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
    return { G, g };
  }

  // 以訓練資料平均/標準差將 X 標準化
  function standardize(X, mu, sd) {
    const p = X[0].length;
    if (!mu) {
      mu = new Float64Array(p); sd = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        const col = X.map(r => r[j]);
        mu[j] = mean(col);
        sd[j] = std(col) || 1;
      }
    }
    const Z = X.map(r => {
      const z = new Float64Array(p);
      for (let j = 0; j < p; j++) z[j] = (r[j] - mu[j]) / sd[j];
      return z;
    });
    return { Z, mu, sd };
  }

  // ---------- 機率分佈 ----------
  function logGamma(x) {
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    x -= 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  function betacf(a, b, x) {
    const FPMIN = 1e-300, EPS = 1e-15;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 500; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  // 正規化不完全 Beta 函數 I_x(a, b)
  function incBeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return bt * betacf(a, b, x) / a;
    return 1 - bt * betacf(b, a, 1 - x) / b;
  }

  // 正規化不完全 Gamma 函數 Q(a, x) = 1 - P(a, x)
  function gammaQ(a, x) {
    if (x <= 0) return 1;
    const gln = logGamma(a);
    if (x < a + 1) {
      let ap = a, sum = 1 / a, del = sum;
      for (let n = 0; n < 500; n++) {
        ap += 1; del *= x / ap; sum += del;
        if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
      }
      return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
    }
    const FPMIN = 1e-300;
    let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
    for (let i = 1; i < 500; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return Math.exp(-x + a * Math.log(x) - gln) * h;
  }

  // t 分佈雙尾 p 值
  function tTwoSided(t, df) {
    return incBeta(df / (df + t * t), df / 2, 0.5);
  }

  function tCdf(t, df) {
    const p = 0.5 * tTwoSided(t, df);
    return t >= 0 ? 1 - p : p;
  }

  function tPpf(q, df) {
    let lo = -1e6, hi = 1e6;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (tCdf(mid, df) < q) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  // F 分佈右尾機率
  function fSf(f, d1, d2) {
    if (!(f > 0)) return 1;
    return incBeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
  }

  function chi2Sf(x, k) {
    return gammaQ(k / 2, x / 2);
  }

  // 標準常態分位數（Acklam 近似）
  function normPpf(p) {
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425;
    if (p < pl) {
      const q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - pl) return -normPpf(1 - p);
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  // ---------- OLS ----------
  // X：n×p（原始尺度），內部以標準化設計矩陣求解以避免數值不穩，係數再轉回原始尺度
  function fitOLS(X, y, names) {
    const n = X.length, p = names.length, k = p + 1;
    const dfResid = n - k;
    if (dfResid <= 0) throw new Error(`樣本數 (${n}) 不足以估計 ${k} 個參數`);
    const { Z: Zs, mu, sd } = standardize(X);
    const Z = Zs.map(z => { const r = new Float64Array(k); r[0] = 1; r.set(z, 1); return r; });
    const { G, g } = crossProducts(Z, y);
    const inv = invert(G);
    if (!inv) throw new Error('設計矩陣奇異：特徵之間存在完全共線，請改用特徵選擇或排除重複欄位');
    const bz = matVec(inv, g);

    const fitted = new Float64Array(n), resid = new Float64Array(n);
    const yBar = mean(y);
    let sse = 0, sst = 0;
    for (let i = 0; i < n; i++) {
      let f = 0;
      for (let j = 0; j < k; j++) f += Z[i][j] * bz[j];
      fitted[i] = f;
      resid[i] = y[i] - f;
      sse += resid[i] ** 2;
      sst += (y[i] - yBar) ** 2;
    }
    const sigma2 = sse / dfResid;
    const covZ = inv.map(row => row.map(v => v * sigma2));

    // 轉換矩陣 T：β_原始 = T · β_標準化
    const T = Array.from({ length: k }, () => new Float64Array(k));
    T[0][0] = 1;
    for (let j = 0; j < p; j++) {
      T[0][j + 1] = -mu[j] / sd[j];
      T[j + 1][j + 1] = 1 / sd[j];
    }
    const coef = matVec(T, bz);
    const TC = T.map(row => covZ.map((_, c) => { let s = 0; for (let m = 0; m < k; m++) s += row[m] * covZ[m][c]; return s; }));
    const cov = TC.map(row => T.map(trow => { let s = 0; for (let m = 0; m < k; m++) s += row[m] * trow[m]; return s; }));

    const se = cov.map((row, i) => Math.sqrt(Math.max(row[i], 0)));
    const tStat = coef.map((b, i) => b / se[i]);
    const pVal = tStat.map(t => tTwoSided(t, dfResid));
    const t975 = tPpf(0.975, dfResid);

    const r2 = 1 - sse / sst;
    const adjR2 = 1 - (1 - r2) * (n - 1) / dfResid;
    const fStat = p > 0 ? ((sst - sse) / p) / sigma2 : NaN;
    const fP = p > 0 ? fSf(fStat, p, dfResid) : NaN;
    const llf = -n / 2 * (Math.log(2 * Math.PI) + Math.log(sse / n) + 1);

    // Breusch–Pagan（Koenker 版）：e² 對同一設計矩陣回歸，LM = n·R²
    const e2 = Array.from(resid, r => r * r);
    const gAux = new Float64Array(k);
    for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) gAux[j] += Z[i][j] * e2[i];
    const bAux = matVec(inv, gAux);
    const e2Bar = mean(e2);
    let sseAux = 0, sstAux = 0;
    for (let i = 0; i < n; i++) {
      let f = 0;
      for (let j = 0; j < k; j++) f += Z[i][j] * bAux[j];
      sseAux += (e2[i] - f) ** 2;
      sstAux += (e2[i] - e2Bar) ** 2;
    }
    const bpLM = n * (1 - sseAux / sstAux);

    function predict(Xnew, alpha = 0.05) {
      const tc = tPpf(1 - alpha / 2, dfResid);
      return Xnew.map(row => {
        const z = new Float64Array(k);
        z[0] = 1;
        for (let j = 0; j < p; j++) z[j + 1] = (row[j] - mu[j]) / sd[j];
        let m = 0;
        for (let j = 0; j < k; j++) m += z[j] * bz[j];
        const vMean = quadForm(covZ, z);
        const seMean = Math.sqrt(vMean), seObs = Math.sqrt(vMean + sigma2);
        return {
          mean: m,
          ciLower: m - tc * seMean, ciUpper: m + tc * seMean,
          piLower: m - tc * seObs, piUpper: m + tc * seObs,
        };
      });
    }

    const sdY = std(y);
    return {
      names: ['const', ...names],
      n, p, dfResid, coef, se, tStat, pVal,
      ciLow: coef.map((b, i) => b - t975 * se[i]),
      ciHigh: coef.map((b, i) => b + t975 * se[i]),
      stdCoef: Array.from(bz.slice(1), b => b / sdY),
      fitted, resid, sigma2, r2, adjR2, fStat, fP,
      aic: -2 * llf + 2 * k,
      bic: -2 * llf + Math.log(n) * k,
      diagnostics: Object.assign(residualDiagnostics(resid), {
        bpLM, bpP: p > 0 ? chi2Sf(bpLM, p) : NaN,
      }),
      predict,
    };
  }

  function residualDiagnostics(resid) {
    const n = resid.length;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      den += resid[i] ** 2;
      if (i > 0) num += (resid[i] - resid[i - 1]) ** 2;
    }
    const m = mean(resid);
    let m2 = 0, m3 = 0, m4 = 0;
    for (let i = 0; i < n; i++) {
      const d = resid[i] - m;
      m2 += d * d; m3 += d ** 3; m4 += d ** 4;
    }
    m2 /= n; m3 /= n; m4 /= n;
    const skew = m3 / m2 ** 1.5, kurt = m4 / (m2 * m2);
    const jb = n / 6 * (skew ** 2 + (kurt - 3) ** 2 / 4);
    return { durbinWatson: num / den, skew, kurtosis: kurt, jb, jbP: chi2Sf(jb, 2) };
  }

  // scipy.stats.probplot 的理論分位數（Filliben 位置）
  function qqPoints(resid) {
    const n = resid.length;
    const sorted = Float64Array.from(resid).sort();
    const theo = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u;
      if (i === n - 1) u = Math.pow(0.5, 1 / n);
      else if (i === 0) u = 1 - Math.pow(0.5, 1 / n);
      else u = (i + 1 - 0.3175) / (n + 0.365);
      theo[i] = normPpf(u);
    }
    return { theoretical: theo, sample: sorted };
  }

  // ---------- 共線性 ----------
  // VIF_j = diag(R⁻¹)_jj，R 為特徵相關矩陣（等價於 statsmodels 的輔助迴歸 1/(1-R²_j)）
  function vif(X) {
    const p = X[0].length;
    if (p === 1) return [1];
    const { Z } = standardize(X);
    const { G } = crossProducts(Z, new Float64Array(Z.length));
    const R = G.map(row => Array.from(row, v => v / Z.length));
    const inv = invert(R);
    return inv ? inv.map((row, i) => row[i]) : Array(p).fill(Infinity);
  }

  // 偵測完全共線欄位：依序以遞增 Cholesky 計算各欄對前面已保留欄位回歸後的殘差變異，
  // 近乎 0 者（可由其他欄線性組合而成）回報 { index, basis }
  function dependentColumns(X, tol = 1e-9) {
    const p = X[0].length, n = X.length;
    const { Z } = standardize(X);
    const { G } = crossProducts(Z, new Float64Array(n));
    const R = G.map(row => Array.from(row, v => v / n));
    const kept = [], L = [], dropped = [];
    for (let j = 0; j < p; j++) {
      const w = [];
      for (let a = 0; a < kept.length; a++) {
        let s = R[kept[a]][j];
        for (let b = 0; b < a; b++) s -= L[a][b] * w[b];
        w.push(s / L[a][a]);
      }
      const resid = R[j][j] - w.reduce((s, v) => s + v * v, 0);
      if (resid < tol) {
        // 回代 Lᵀβ = w 得到以已保留欄位表示此欄的迴歸係數
        const beta = new Array(kept.length).fill(0);
        for (let a = kept.length - 1; a >= 0; a--) {
          let s = w[a];
          for (let c = a + 1; c < kept.length; c++) s -= L[c][a] * beta[c];
          beta[a] = s / L[a][a];
        }
        dropped.push({ index: j, basis: kept.filter((_, a) => Math.abs(beta[a]) > 1e-6) });
      } else {
        L.push([...w, Math.sqrt(resid)]);
        kept.push(j);
      }
    }
    return dropped;
  }

  // ---------- 特徵選擇 ----------
  // f_regression：F = r²/(1-r²)·(n-2)
  function fRegression(X, y) {
    const n = X.length, p = X[0].length;
    const out = [];
    for (let j = 0; j < p; j++) {
      const r = pearson(X.map(row => row[j]), y);
      const F = r * r / Math.max(1 - r * r, 1e-300) * (n - 2);
      out.push({ F, r, p: fSf(F, 1, n - 2) });
    }
    return out;
  }

  function topK(scores, k) {
    return scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(d => d[1]).sort((a, b) => a - b);
  }

  // RFE：以標準化係數絕對值為重要度，每輪剔除最弱特徵；rank 1 = 入選
  function rfe(X, y, k) {
    const p = X[0].length;
    const { Z } = standardize(X);
    const yBar = mean(y);
    const yc = Array.from(y, v => v - yBar);
    const { G, g } = crossProducts(Z, yc);
    const active = Array.from({ length: p }, (_, i) => i);
    const rank = new Array(p).fill(1);
    let nextRank = p - k + 1;
    while (active.length > k) {
      const sub = active.map(a => active.map(b => G[a][b] + (a === b ? 1e-8 : 0)));
      const inv = invert(sub);
      const beta = inv ? matVec(inv, active.map(a => g[a])) : active.map(a => g[a]);
      let worst = 0;
      for (let i = 1; i < active.length; i++) if (Math.abs(beta[i]) < Math.abs(beta[worst])) worst = i;
      rank[active[worst]] = nextRank--;
      active.splice(worst, 1);
    }
    return { selected: active.slice(), rank };
  }

  // 座標下降 Lasso：min (1/2n)||y - Xb||² + α||b||₁；G = X'X/n、c = X'y/n（皆已中心化）
  function lassoCD(G, c, alpha, b0) {
    const p = c.length;
    const b = b0 ? Float64Array.from(b0) : new Float64Array(p);
    const Gb = new Float64Array(p);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) Gb[i] += G[i][j] * b[j];
    for (let iter = 0; iter < 2000; iter++) {
      let maxDelta = 0, maxB = 0;
      for (let j = 0; j < p; j++) {
        if (G[j][j] === 0) continue;
        const rho = c[j] - Gb[j] + G[j][j] * b[j];
        const nb = Math.sign(rho) * Math.max(Math.abs(rho) - alpha, 0) / G[j][j];
        const delta = nb - b[j];
        if (delta !== 0) {
          for (let i = 0; i < p; i++) Gb[i] += G[i][j] * delta;
          b[j] = nb;
        }
        maxDelta = Math.max(maxDelta, Math.abs(delta));
        maxB = Math.max(maxB, Math.abs(nb));
      }
      if (maxDelta <= 1e-7 * Math.max(maxB, 1e-12)) break;
    }
    return b;
  }

  function centeredMoments(Z, y, rows) {
    const p = Z[0].length, n = rows.length;
    const mx = new Float64Array(p);
    let my = 0;
    for (const i of rows) { for (let j = 0; j < p; j++) mx[j] += Z[i][j]; my += y[i]; }
    for (let j = 0; j < p; j++) mx[j] /= n;
    my /= n;
    const Zc = rows.map(i => { const r = new Float64Array(p); for (let j = 0; j < p; j++) r[j] = Z[i][j] - mx[j]; return r; });
    const yc = rows.map(i => y[i] - my);
    const { G, g } = crossProducts(Zc, yc);
    return {
      G: G.map(row => Array.from(row, v => v / n)),
      c: Array.from(g, v => v / n),
      mx, my,
    };
  }

  // LassoCV：標準化後 5-fold（不打亂，同 sklearn KFold 預設）挑選 α
  function lassoCV(X, y, folds = 5, nAlphas = 50) {
    const n = X.length, p = X[0].length;
    const { Z } = standardize(X);
    const all = Array.from({ length: n }, (_, i) => i);
    const full = centeredMoments(Z, y, all);
    const alphaMax = Math.max(...full.c.map(Math.abs));
    const alphas = Array.from({ length: nAlphas }, (_, i) => alphaMax * Math.pow(1e-3, i / (nAlphas - 1)));

    const mse = new Array(nAlphas).fill(0);
    for (let f = 0; f < folds; f++) {
      const lo = Math.floor(f * n / folds), hi = Math.floor((f + 1) * n / folds);
      const trainRows = all.filter(i => i < lo || i >= hi);
      const m = centeredMoments(Z, y, trainRows);
      let b = null;
      alphas.forEach((a, ai) => {
        b = lassoCD(m.G, m.c, a, b);
        let s = 0;
        for (let i = lo; i < hi; i++) {
          let pred = m.my;
          for (let j = 0; j < p; j++) pred += (Z[i][j] - m.mx[j]) * b[j];
          s += (y[i] - pred) ** 2;
        }
        mse[ai] += s / (hi - lo) / folds;
      });
    }
    let best = 0;
    for (let i = 1; i < nAlphas; i++) if (mse[i] < mse[best]) best = i;
    let b = null;
    for (let i = 0; i <= best; i++) b = lassoCD(full.G, full.c, alphas[i], b);
    const coef = Array.from(b);
    return {
      alphas, mse, alpha: alphas[best], coef,
      selected: coef.map((v, i) => [v, i]).filter(d => Math.abs(d[0]) > 1e-10).map(d => d[1]),
    };
  }

  // ---------- 評估指標 ----------
  function regressionMetrics(yTrue, yPred, p) {
    const n = yTrue.length;
    const yBar = mean(yTrue);
    let sse = 0, sae = 0, sst = 0;
    for (let i = 0; i < n; i++) {
      const e = yTrue[i] - yPred[i];
      sse += e * e; sae += Math.abs(e); sst += (yTrue[i] - yBar) ** 2;
    }
    const r2 = 1 - sse / sst;
    return {
      mse: sse / n, rmse: Math.sqrt(sse / n), mae: sae / n, r2,
      adjR2: n - p - 1 > 0 ? 1 - (1 - r2) * (n - 1) / (n - p - 1) : r2,
    };
  }

  const api = {
    mulberry32, trainTestSplit, mean, std, median, quantile, pearson,
    invert, standardize, tPpf, tTwoSided, fSf, chi2Sf, normPpf,
    fitOLS, residualDiagnostics, qqPoints, vif, dependentColumns, fRegression, topK, rfe, lassoCV, regressionMetrics,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Stats = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
