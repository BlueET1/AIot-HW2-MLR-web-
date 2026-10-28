/*
 * app.js — 介面與 CRISP-DM 各階段呈現。
 * 資料於瀏覽器端解析、建模，因此可直接部署於 GitHub Pages（無後端）。
 */
(function () {
  'use strict';

  const S = window.Stats;
  const $ = id => document.getElementById(id);

  const DEFAULT_URL = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv';

  const PROBLEMS = {
    p1: {
      name: 'WHO Life Expectancy（全球各國預期壽命）',
      file: 'data/Life_Expectancy_Data.csv',
      target: 'Life expectancy',
      ignore: ['Country'],
      source: 'https://www.kaggle.com/datasets/kumarrajarshi/life-expectancy-who',
      business: `
        <p><strong>背景：</strong>世界衛生組織 (WHO) 與聯合國彙整 2000–2015 年 193 個國家的健康、疫苗、死亡率與社會經濟指標。
        各國衛生預算有限，需要知道哪些因素與「國民平均預期壽命」關聯最強，才能決定資源優先順序。</p>
        <p><strong>業務問題：</strong>成人死亡率、HIV/AIDS 盛行率、疫苗覆蓋率、受教育年數、人均 GDP 等因素，各自對預期壽命的邊際影響有多大？
        給定一國的指標組合，其預期壽命的合理範圍為何？</p>
        <p><strong>利害關係人：</strong>各國衛生部門、國際援助組織 (WHO / UNICEF)、公共衛生研究者。</p>`,
      success: [
        '測試集 R² ≥ 0.80、RMSE < 4 歲',
        '回歸係數方向符合公衛常識（例如成人死亡率為負、教育年數為正）',
        '95% 預測區間實際涵蓋率接近 95%',
        '經特徵選擇後共線性（VIF）明顯下降、模型更精簡',
      ],
    },
    p2: {
      name: 'King County House Sales（西雅圖國王郡房價）',
      file: 'data/kc_house_data.csv',
      target: 'price',
      ignore: [],
      source: 'https://www.kaggle.com/datasets/harlfoxem/housesalesprediction',
      business: `
        <p><strong>背景：</strong>美國華盛頓州 King County（含西雅圖）2014–2015 年房屋成交紀錄，包含坪數、房間數、屋況、建造等級、地理位置等屬性。</p>
        <p><strong>業務問題：</strong>哪些房屋屬性對成交價影響最大？給定一間房屋的條件，能否估計出合理的成交價與其不確定範圍，協助議價與貸款鑑價？</p>
        <p><strong>利害關係人：</strong>購屋者與賣方、房仲業者、不動產估價師、承作房貸的銀行。</p>`,
      success: [
        '測試集 R² ≥ 0.70',
        '找出統計顯著的關鍵價格驅動因子（p < 0.05）',
        '以預測區間呈現單一房屋的價格範圍，涵蓋率接近名目水準',
      ],
    },
  };

  const METHOD_LABEL = { all: '全特徵', rfe: 'RFE', kbest: 'SelectKBest', lasso: 'LassoCV' };

  const state = {
    problem: 'p1',
    cache: {},
    table: null,
    source: '',
    target: null,
    ignore: [],
    k: 14,
    R: null,
    activeTab: 'business',
    dirty: new Set(),
  };

  // ---------- 工具 ----------
  function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function fmt(v, d = 4) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    if (!Number.isFinite(v)) return '∞';
    const a = Math.abs(v);
    if (a !== 0 && a < 1e-3) return v.toExponential(2);
    if (a >= 1e12) return v.toExponential(3);
    if (a >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 2 });
    return v.toFixed(d);
  }

  function fmtP(p) {
    if (!Number.isFinite(p)) return '—';
    if (p < 1e-16) return '< 1e-16';
    return p < 1e-4 ? p.toExponential(2) : p.toFixed(4);
  }

  function setStatus(msg, isError) {
    const el = $('status');
    el.textContent = msg;
    el.classList.toggle('error', !!isError);
  }

  function table(headers, rows, opts = {}) {
    const head = headers.map(h => `<th>${h}</th>`).join('');
    const body = rows.map((r, i) => `<tr${opts.rowClass && opts.rowClass(i) ? ` class="${opts.rowClass(i)}"` : ''}>${r.map((c, j) => {
      const cls = opts.cellClass ? opts.cellClass(i, j) : '';
      return `<td${cls ? ` class="${cls}"` : ''}>${c}</td>`;
    }).join('')}</tr>`).join('');
    return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function metric(label, value, sub = '') {
    return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;
  }

  // 與 style.css 的 --dp-color-* 一致：鉛筆墨色、藍色註記、紅色修正
  const COLORS = { ink: '#38352f', blue: '#344c69', blueBand: 'rgba(52,76,105,0.38)', blueWide: 'rgba(52,76,105,0.13)', red: '#9e3a2e', grid: 'rgba(179,172,157,0.55)', paper: '#fcf8ef' };

  function plot(id, traces, layout = {}) {
    const base = {
      margin: { l: 64, r: 20, t: 52, b: 104 },
      font: { family: getComputedStyle(document.body).fontFamily, size: 15, color: COLORS.ink },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: { gridcolor: COLORS.grid, zeroline: false, automargin: true },
      yaxis: { gridcolor: COLORS.grid, zeroline: false, automargin: true },
      // 圖例固定在整張圖底部，避免與 x 軸標題重疊
      legend: { orientation: 'h', yref: 'container', y: 0, yanchor: 'bottom', x: 0, font: { size: 14 } },
    };
    const merged = Object.assign({}, base, layout, {
      xaxis: Object.assign({}, base.xaxis, layout.xaxis),
      yaxis: Object.assign({}, base.yaxis, layout.yaxis),
    });
    Plotly.react(id, traces, merged, { responsive: true, displaylogo: false });
  }

  // 大量點改以等距抽樣呈現（不使用 WebGL，避免部分瀏覽器不支援）
  const thin = (a, max = 4000) => {
    const arr = Array.from(a);
    const step = Math.ceil(arr.length / max);
    return step > 1 ? arr.filter((_, i) => i % step === 0) : arr;
  };

  // ---------- 資料載入 ----------
  function parseCSV(text) {
    const res = Papa.parse(text.trim(), { header: true, skipEmptyLines: true, transformHeader: h => h.trim() });
    const columns = res.meta.fields.filter(c => c !== '');
    const NA = new Set(['', 'na', 'nan', 'null', 'none', '?']);
    const rows = res.data.map(r => {
      const o = {};
      for (const c of columns) {
        const s = r[c] == null ? '' : String(r[c]).trim();
        if (NA.has(s.toLowerCase())) o[c] = null;
        else {
          const v = Number(s);
          o[c] = Number.isFinite(v) ? v : s;
        }
      }
      return o;
    });
    const types = {};
    for (const c of columns) {
      const vals = rows.map(r => r[c]).filter(v => v !== null);
      types[c] = vals.length && vals.every(v => typeof v === 'number') ? 'num' : 'cat';
    }
    if (!rows.length || !columns.length) throw new Error('CSV 內容為空或格式無法解析');
    return { columns, rows, types };
  }

  async function fetchCSV(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}：無法下載 ${url}`);
    return parseCSV(await resp.text());
  }

  async function loadProblem(key) {
    const P = PROBLEMS[key];
    setStatus('載入資料中…');
    if (!state.cache[key]) state.cache[key] = await fetchCSV(P.file);
    state.table = state.cache[key];
    state.source = P.source;
    state.target = P.target;
    state.ignore = P.ignore.slice();
    state.k = 14;
    recompute();
  }

  function useCustomTable(tbl, source) {
    const numeric = tbl.columns.filter(c => tbl.types[c] === 'num');
    if (!numeric.length) throw new Error('找不到可作為目標變數的數值欄位');
    state.table = tbl;
    state.source = source;
    state.target = numeric[numeric.length - 1];
    state.ignore = [];
    state.k = 14;
    renderCustomControls();
    recompute();
  }

  async function loadCustomUrl() {
    const url = $('csvUrl').value.trim() || DEFAULT_URL;
    setStatus('下載自訂資料集中…');
    try {
      useCustomTable(await fetchCSV(url), url);
    } catch (e) {
      setStatus(`資料載入失敗：${e.message}（若為 CORS 限制，請改用「上傳 CSV」）`, true);
    }
  }

  function renderCustomControls() {
    const tbl = state.table;
    $('target').innerHTML = tbl.columns.filter(c => tbl.types[c] === 'num')
      .map(c => `<option${c === state.target ? ' selected' : ''}>${esc(c)}</option>`).join('');
    $('ignoreList').innerHTML = tbl.columns.filter(c => c !== state.target).map(c => `
      <label><input type="checkbox" value="${esc(c)}"${state.ignore.includes(c) ? ' checked' : ''}> ${esc(c)}
      <span class="hint">(${tbl.types[c] === 'num' ? '數值' : '類別'})</span></label>`).join('');
  }

  // ---------- Data Preparation ----------
  function buildDesign(tbl, target, ignore, testSize, seed) {
    const rows = tbl.rows.filter(r => typeof r[target] === 'number');
    const y = rows.map(r => r[target]);
    const notes = { droppedRows: tbl.rows.length - rows.length, dummies: [], droppedCols: [], imputed: [], constant: [] };

    const cols = [];
    for (const c of tbl.columns) {
      if (c === target || ignore.includes(c)) continue;
      if (tbl.types[c] === 'num') {
        cols.push({ name: c, values: rows.map(r => r[c]) });
        continue;
      }
      const levels = [...new Set(rows.map(r => r[c]).filter(v => v !== null).map(String))].sort();
      if (levels.length >= 2 && levels.length <= 5) {
        // 虛擬變數編碼（drop_first，避免完全共線）
        levels.slice(1).forEach(l => cols.push({ name: `${c}_${l}`, values: rows.map(r => (r[c] !== null && String(r[c]) === l ? 1 : 0)) }));
        notes.dummies.push({ col: c, levels });
      } else {
        notes.droppedCols.push({ col: c, reason: `類別數 ${levels.length}（>5 或僅 1 類），不適合虛擬編碼` });
      }
    }

    // 依原始列順序排列訓練索引，使 Durbin-Watson 反映資料原始次序
    const split = S.trainTestSplit(rows.length, testSize, seed);
    split.train.sort((a, b) => a - b);
    split.test.sort((a, b) => a - b);

    const keep = [];
    for (const col of cols) {
      const trainVals = split.train.map(i => col.values[i]).filter(v => v !== null);
      if (!trainVals.length) {
        notes.droppedCols.push({ col: col.name, reason: '訓練集全為缺失值' });
        continue;
      }
      const missing = col.values.filter(v => v === null).length;
      if (missing) {
        const med = S.median(trainVals);
        col.values = col.values.map(v => (v === null ? med : v));
        notes.imputed.push({ col: col.name, missing, median: med });
      }
      if (S.std(split.train.map(i => col.values[i])) === 0) {
        notes.constant.push(col.name);
        continue;
      }
      keep.push(col);
    }
    if (!keep.length) throw new Error('沒有可用的特徵欄位');

    // 移除可由其他欄位線性組合而成的欄位，否則 X'X 不可逆
    const trainX = split.train.map(i => keep.map(c => c.values[i]));
    const dep = S.dependentColumns(trainX);
    notes.collinear = dep.map(d => ({ col: keep[d.index].name, basis: d.basis.map(b => keep[b].name) }));
    const depSet = new Set(dep.map(d => d.index));
    const final = keep.filter((_, j) => !depSet.has(j));

    return {
      names: final.map(c => c.name),
      X: rows.map((_, i) => final.map(c => c.values[i])),
      y, rows, split, notes,
    };
  }

  const minOf = a => a.reduce((m, v) => (v < m ? v : m), Infinity);
  const maxOf = a => a.reduce((m, v) => (v > m ? v : m), -Infinity);

  const pick = (X, cols) => X.map(r => cols.map(j => r[j]));

  function recompute() {
    if (!state.table) return;
    const tbl = state.table;
    const testSize = +$('testSize').value;
    const seed = parseInt($('seed').value, 10) || 0;
    const alpha = +$('conf').value;
    const method = $('fsMethod').value;
    try {
      const D = buildDesign(tbl, state.target, state.ignore, testSize, seed);
      const p = D.names.length;
      const Xtr = D.split.train.map(i => D.X[i]), ytr = D.split.train.map(i => D.y[i]);
      const Xte = D.split.test.map(i => D.X[i]), yte = D.split.test.map(i => D.y[i]);
      if (Xtr.length < p + 3) throw new Error(`訓練樣本 (${Xtr.length}) 太少，無法估計 ${p} 個特徵`);

      const k = Math.max(1, Math.min(state.k, p));
      const kInput = $('kFeatures');
      kInput.max = p;
      kInput.value = k;
      $('kOut').textContent = k;

      const allIdx = D.names.map((_, i) => i);
      const fScores = S.fRegression(Xtr, ytr);
      const rfeRes = S.rfe(Xtr, ytr, k);
      const lasso = S.lassoCV(Xtr, ytr);
      const sets = {
        all: allIdx,
        rfe: rfeRes.selected,
        kbest: S.topK(fScores.map(f => f.F), k),
        lasso: lasso.selected.length ? lasso.selected : allIdx,
      };

      const comparison = Object.entries(sets).map(([m, sel]) => {
        try {
          const mod = S.fitOLS(pick(Xtr, sel), ytr, sel.map(j => D.names[j]));
          const pr = mod.predict(pick(Xte, sel), alpha).map(r => r.mean);
          return Object.assign({ method: m, n: sel.length, trainR2: mod.r2, aic: mod.aic }, S.regressionMetrics(yte, pr, sel.length));
        } catch (e) {
          return { method: m, n: sel.length, error: e.message };
        }
      });

      const sel = sets[method];
      const selNames = sel.map(j => D.names[j]);
      const model = S.fitOLS(pick(Xtr, sel), ytr, selNames);
      const preds = model.predict(pick(Xte, sel), alpha);
      const metrics = S.regressionMetrics(yte, preds.map(r => r.mean), sel.length);
      const coverage = preds.filter((r, i) => yte[i] >= r.piLower && yte[i] <= r.piUpper).length / yte.length;
      const ciCoverage = preds.filter((r, i) => yte[i] >= r.ciLower && yte[i] <= r.ciUpper).length / yte.length;

      state.R = {
        D, p, k, alpha, method, sets, sel, selNames, fScores, rfeRes, lasso, comparison,
        Xtr, ytr, Xte, yte, model, preds, metrics, coverage, ciCoverage,
        vif: S.vif(pick(Xtr, sel)),
      };
      $('kBox').hidden = !(method === 'rfe' || method === 'kbest');
      setStatus(`已完成建模：${D.rows.length.toLocaleString()} 筆樣本、${p} 個特徵，${METHOD_LABEL[method]} 選出 ${sel.length} 個。`);
      ['business', 'data', 'prep', 'model', 'eval', 'deploy', 'ai'].forEach(t => state.dirty.add(t));
      renderTab(state.activeTab);
    } catch (e) {
      console.error(e);
      setStatus(`分析失敗：${e.message}`, true);
    }
  }

  // ---------- 分頁呈現 ----------
  function renderTab(name) {
    if (!state.R || !state.dirty.has(name)) return;
    state.dirty.delete(name);
    RENDER[name]();
  }

  const datasetTitle = () => (state.problem === 'custom' ? '自訂資料集' : PROBLEMS[state.problem].name);
  const confLabel = () => `${Math.round((1 - state.R.alpha) * 100)}%`;

  const RENDER = {
    business() {
      const P = PROBLEMS[state.problem];
      const body = P ? P.business : `
        <p><strong>背景：</strong>使用者自行提供的資料集，示範本系統可泛用於任意表格資料的多元線性回歸分析。</p>
        <p><strong>業務問題：</strong>以其餘欄位預測 <code>${esc(state.target)}</code>，並量化各特徵的邊際效應與預測不確定性。</p>`;
      const success = P ? P.success : ['測試集 R² 越高越好，並與全特徵模型比較', '預測區間涵蓋率接近名目信心水準'];
      $('tab-business').innerHTML = `
        <h2>階段一：Business Understanding（商業理解）</h2>
        <ol class="process" aria-label="CRISP-DM 流程">${['Business Understanding', 'Data Understanding', 'Data Preparation', 'Modeling', 'Evaluation', 'Deployment'].map((s, i) => `<li>${s}${i < 5 ? '<span class="arrow" aria-hidden="true">⟶</span>' : ''}</li>`).join('')}</ol>
        <div class="card">
          <h3 style="margin-top:0">${esc(datasetTitle())}</h3>
          ${body}
          <p><strong>預測目標 (Target)：</strong><code>${esc(state.target)}</code>（連續數值 → 回歸問題）</p>
          <p><strong>資料來源：</strong><a href="${esc(state.source)}" target="_blank" rel="noopener">${esc(state.source)}</a></p>
        </div>
        <div class="card">
          <h3 style="margin-top:0">分析方法與成功標準</h3>
          <p>以 <strong>多元線性回歸 (OLS)</strong> 建模，比較全特徵、RFE、SelectKBest 與 LassoCV 四種特徵選擇策略，
          並以 ${confLabel()} <strong>信賴區間 (CI)</strong> 與 <strong>預測區間 (PI)</strong> 呈現預測不確定性。</p>
          <ul>${success.map(s => `<li>${s}</li>`).join('')}</ul>
        </div>`;
    },

    data() {
      const R = state.R, tbl = state.table, D = R.D;
      const target = state.target;
      const yAll = D.y;
      const featureCountOk = R.p > 13;

      const describe = tbl.columns.map(c => {
        const vals = tbl.rows.map(r => r[c]);
        const nonNull = vals.filter(v => v !== null);
        const missing = vals.length - nonNull.length;
        if (tbl.types[c] === 'num') {
          return [esc(c), '數值', nonNull.length, missing, fmt(S.mean(nonNull), 2), fmt(S.std(nonNull, 1), 2),
            fmt(minOf(nonNull), 2), fmt(S.median(nonNull), 2), fmt(maxOf(nonNull), 2)];
        }
        return [esc(c), '類別', nonNull.length, missing, `${new Set(nonNull).size} 類`, '', '', '', ''];
      });

      const preview = tbl.rows.slice(0, 10).map(r => tbl.columns.map(c => (r[c] === null ? '<span class="hint">NaN</span>' : esc(typeof r[c] === 'number' ? fmt(r[c], 2) : r[c]))));

      $('tab-data').innerHTML = `
        <h2>階段二：Data Understanding（資料理解）</h2>
        <div class="metrics">
          ${metric('樣本數 (Rows)', tbl.rows.length.toLocaleString())}
          ${metric('原始欄位數', tbl.columns.length)}
          ${metric('編碼後輸入特徵數', R.p, featureCountOk ? '<span class="badge good">符合 &gt; 13 特徵</span>' : '<span class="badge warn">少於 14 個特徵</span>')}
          ${metric(`目標 ${esc(target)} 平均`, fmt(S.mean(yAll), 2), `標準差 ${fmt(S.std(yAll, 1), 2)}`)}
        </div>
        <div class="card"><h3 style="margin-top:0">資料預覽（前 10 筆）</h3>
          ${table(tbl.columns.map(esc), preview)}</div>
        <div class="card"><h3 style="margin-top:0">欄位敘述統計與缺失值</h3>
          ${table(['欄位', '型態', '非缺失', '缺失', '平均', '標準差', '最小', '中位數', '最大'], describe, {
            cellClass: (i, j) => (j === 3 && describe[i][3] > 0 ? 'cell-warn' : ''),
          })}</div>
        <div class="grid-2">
          <div class="card"><div id="plotTarget" class="plot"></div></div>
          <div class="card"><div id="plotTargetCorr" class="plot"></div></div>
        </div>
        <div class="card"><div id="plotCorr" class="plot" style="min-height:${Math.max(480, 22 * (R.p + 1))}px"></div></div>`;

      plot('plotTarget', [{ type: 'histogram', x: yAll, nbinsx: 40, marker: { color: COLORS.blue, line: { color: COLORS.paper, width: 1 } } }],
        { title: `目標變數 ${target} 分佈`, xaxis: { title: target }, yaxis: { title: '次數' }, bargap: 0.02 });

      const corrT = D.names.map((n, j) => ({ n, r: S.pearson(D.X.map(row => row[j]), yAll) })).sort((a, b) => Math.abs(a.r) - Math.abs(b.r));
      plot('plotTargetCorr', [{
        type: 'bar', orientation: 'h', y: corrT.map(d => d.n), x: corrT.map(d => d.r),
        marker: { color: corrT.map(d => (d.r >= 0 ? COLORS.blue : COLORS.red)) },
        hovertemplate: '%{y}: r = %{x:.3f}<extra></extra>',
      }], { title: `各特徵與 ${target} 的 Pearson 相關`, xaxis: { title: 'r', range: [-1, 1] }, margin: { l: 170, r: 20, t: 48, b: 48 }, height: Math.max(380, 20 * R.p + 100) });

      const names = [...D.names, target];
      const cols = [...D.names.map((_, j) => D.X.map(row => row[j])), yAll];
      const M = cols.map(a => cols.map(b => S.pearson(a, b)));
      plot('plotCorr', [{
        type: 'heatmap', z: M, x: names, y: names, zmin: -1, zmax: 1, colorscale: [[0, COLORS.red], [0.5, '#f7f2e7'], [1, COLORS.blue]],
        text: names.length <= 22 ? M.map(r => r.map(v => v.toFixed(2))) : null, texttemplate: names.length <= 22 ? '%{text}' : '',
        hovertemplate: '%{y} × %{x}: %{z:.3f}<extra></extra>',
      }], { title: '特徵相關係數熱力圖（含目標變數）', yaxis: { autorange: 'reversed' }, margin: { l: 170, r: 20, t: 48, b: 150 } });
    },

    prep() {
      const R = state.R, D = R.D, N = D.notes;
      const steps = [
        `欄位名稱去除前後空白；<code>NA / NaN / 空字串</code> 視為缺失值。`,
        `移除目標變數 <code>${esc(state.target)}</code> 缺失的樣本：${N.droppedRows} 筆。`,
        `排除非預測欄位：${state.ignore.length ? state.ignore.map(c => `<code>${esc(c)}</code>`).join('、') : '無'}。`,
        N.dummies.length
          ? `類別特徵虛擬編碼 (drop_first)：${N.dummies.map(d => `<code>${esc(d.col)}</code>（${d.levels.map(esc).join(' / ')}，基準類別 ${esc(d.levels[0])}）`).join('；')}。`
          : '無類別特徵需要編碼。',
        N.droppedCols.length ? `捨棄欄位：${N.droppedCols.map(d => `<code>${esc(d.col)}</code>（${d.reason}）`).join('；')}。` : '無需捨棄的欄位。',
        `依測試集比例 ${$('testSize').value} 隨機切分（seed = ${esc($('seed').value)}）：訓練 ${R.Xtr.length} 筆、測試 ${R.Xte.length} 筆。`,
        N.imputed.length
          ? `以<strong>訓練集中位數</strong>填補缺失值（避免資料洩漏）：${N.imputed.map(d => `<code>${esc(d.col)}</code> ${d.missing} 筆`).join('、')}。`
          : '特徵無缺失值。',
        N.constant.length ? `移除訓練集零變異欄位：${N.constant.map(esc).join('、')}。` : '',
        ...N.collinear.map(d => `移除完全共線欄位 <code>${esc(d.col)}</code>：可由 ${d.basis.map(b => `<code>${esc(b)}</code>`).join('、')} 線性組合而成，保留會使 X'X 不可逆。`),
      ].filter(Boolean);

      const selSet = new Set(R.sel);
      const kbestSet = new Set(R.sets.kbest);
      const fRank = R.fScores.map((f, i) => [f.F, i]).sort((a, b) => b[0] - a[0]).reduce((acc, [, i], r) => (acc[i] = r + 1, acc), {});
      const fsRows = D.names.map((n, j) => [
        esc(n), fmt(R.fScores[j].r, 3), fmt(R.fScores[j].F, 1), `${fRank[j]}${kbestSet.has(j) ? ' ✓' : ''}`,
        `${R.rfeRes.rank[j]}${R.rfeRes.rank[j] === 1 ? ' ✓' : ''}`,
        `${fmt(R.lasso.coef[j], 4)}${Math.abs(R.lasso.coef[j]) > 1e-10 ? ' ✓' : ''}`,
        selSet.has(j) ? '<span class="badge good">入選</span>' : '<span class="hint">—</span>',
      ]);

      const vifRows = R.selNames.map((n, i) => [esc(n), fmt(R.vif[i], 2)]).sort((a, b) => parseFloat(b[1].replace(/,/g, '')) - parseFloat(a[1].replace(/,/g, '')));
      const vifClass = i => {
        const v = parseFloat(vifRows[i][1].replace(/,/g, ''));
        return v > 10 ? 'cell-bad' : v > 5 ? 'cell-warn' : 'cell-good';
      };

      $('tab-prep').innerHTML = `
        <h2>階段三：Data Preparation（資料準備與特徵選擇）</h2>
        <div class="card"><h3 style="margin-top:0">前處理步驟</h3><ol>${steps.map(s => `<li>${s}</li>`).join('')}</ol></div>
        <div class="card">
          <h3 style="margin-top:0">特徵選擇比較（僅使用訓練集）</h3>
          <p>目前方法：<strong>${METHOD_LABEL[R.method]}</strong>，從 <strong>${R.p}</strong> 個特徵中選出 <strong>${R.sel.length}</strong> 個。
          SelectKBest 依單變量 F 統計量排序；RFE 每輪以標準化係數絕對值剔除最弱特徵；LassoCV 以 5-fold 交叉驗證選出 α = ${fmt(R.lasso.alpha, 5)}，係數歸零者即被淘汰。✓ 表示該方法入選（k = ${R.k}）。</p>
          ${table(['特徵', 'r (與目標)', 'F 值', 'F 排名', 'RFE 排名', 'Lasso 係數 (標準化)', '目前模型'], fsRows)}
        </div>
        <div class="grid-2">
          <div class="card"><div id="plotLasso" class="plot"></div></div>
          <div class="card">
            <h3 style="margin-top:0">共線性診斷 VIF（入選特徵）</h3>
            ${table(['特徵', 'VIF'], vifRows, { cellClass: (i, j) => (j === 1 ? vifClass(i) : '') })}
            <p class="hint">VIF &gt; 10 嚴重共線（紅）、5–10 中度（黃）、&lt; 5 良好（綠）。</p>
          </div>
        </div>`;

      plot('plotLasso', [{
        type: 'scatter', mode: 'lines+markers', x: R.lasso.alphas, y: R.lasso.mse, line: { color: COLORS.blue }, marker: { size: 4 }, name: 'CV MSE',
      }, {
        type: 'scatter', mode: 'markers', x: [R.lasso.alpha], y: [minOf(R.lasso.mse)], marker: { color: COLORS.red, size: 11, symbol: 'diamond' }, name: '最佳 α',
      }], { title: 'LassoCV：5-fold 交叉驗證 MSE vs α', xaxis: { title: 'α (log)', type: 'log' }, yaxis: { title: 'CV MSE' } });
    },

    model() {
      const R = state.R, M = R.model, dg = M.diagnostics;
      const stars = p => (p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : '');
      const coefRows = M.names.map((n, i) => [esc(n), fmt(M.coef[i]), fmt(M.se[i]), fmt(M.tStat[i], 3), `${fmtP(M.pVal[i])} ${stars(M.pVal[i])}`, fmt(M.ciLow[i]), fmt(M.ciHigh[i])]);
      const maxVif = maxOf(R.vif);

      const badge = (ok, okText, warnText, bad) => `<span class="badge ${ok ? 'good' : bad ? 'bad' : 'warn'}">${ok ? okText : warnText}</span>`;
      const assumptions = [
        ['1. 線性 (Linearity)', '殘差 vs 預測值圖', '目視：殘差應隨機分佈於 0 附近，無明顯曲線', '<span class="badge warn">目視檢查</span>'],
        ['2. 常態性 (Normality)', `Jarque–Bera = ${fmt(dg.jb, 2)}`, `p = ${fmtP(dg.jbP)}；偏態 ${fmt(dg.skew, 3)}、峰度 ${fmt(dg.kurtosis, 3)}`,
          badge(dg.jbP > 0.05, '符合', '偏離常態（大樣本下依 CLT 推論仍近似有效）')],
        ['3. 同質變異 (Homoscedasticity)', `Breusch–Pagan LM = ${fmt(dg.bpLM, 2)}`, `p = ${fmtP(dg.bpP)}`,
          badge(dg.bpP > 0.05, '符合', '存在異質變異（可考慮 log 轉換或穩健標準誤）')],
        ['4. 獨立性 (Independence)', `Durbin–Watson = ${fmt(dg.durbinWatson, 3)}`, '依原始資料順序計算；≈ 2 表示無一階自相關',
          badge(dg.durbinWatson > 1.5 && dg.durbinWatson < 2.5, '符合', '殘差具自相關（資料具時間/群組結構）')],
        ['5. 無嚴重共線性 (No Multicollinearity)', `最大 VIF = ${fmt(maxVif, 2)}`, '各特徵 VIF 見階段三',
          badge(maxVif < 5, '符合', maxVif > 10 ? '嚴重共線' : '中度共線', maxVif > 10)],
      ];

      $('tab-model').innerHTML = `
        <h2>階段四：Modeling（OLS 多元線性回歸與假設檢定）</h2>
        <div class="metrics">
          ${metric('訓練樣本 n', M.n.toLocaleString())}
          ${metric('特徵數 p', M.p, METHOD_LABEL[R.method])}
          ${metric('R² (訓練)', fmt(M.r2))}
          ${metric('Adj. R² (訓練)', fmt(M.adjR2))}
          ${metric('F 統計量', fmt(M.fStat, 2), `p = ${fmtP(M.fP)}`)}
          ${metric('AIC / BIC', fmt(M.aic, 1), `BIC ${fmt(M.bic, 1)}`)}
          ${metric('殘差標準誤 σ̂', fmt(Math.sqrt(M.sigma2)))}
        </div>
        <div class="card">
          <h3 style="margin-top:0">迴歸係數表（原始尺度）</h3>
          <p class="hint">模型：${esc(state.target)} = β₀ + Σ βⱼ·xⱼ + ε；顯著性 *** p&lt;0.001、** p&lt;0.01、* p&lt;0.05；區間為 95% 係數信賴區間。</p>
          ${table(['變數', '係數 β', '標準誤', 't', 'p 值', '[0.025', '0.975]'], coefRows)}
        </div>
        <div class="card"><div id="plotStdCoef" class="plot"></div></div>
        <div class="card">
          <h3 style="margin-top:0">迴歸五大假設檢定</h3>
          ${table(['假設', '檢定統計量', '說明', '結果'], assumptions, { cellClass: (i, j) => (j === 2 ? 'text' : '') })}
        </div>
        <div class="grid-2">
          <div class="card"><div id="plotResid" class="plot"></div></div>
          <div class="card"><div id="plotQQ" class="plot"></div></div>
        </div>
        <div class="card"><div id="plotResidHist" class="plot"></div></div>`;

      const order = M.stdCoef.map((v, i) => [v, R.selNames[i]]).sort((a, b) => Math.abs(a[0]) - Math.abs(b[0]));
      plot('plotStdCoef', [{
        type: 'bar', orientation: 'h', x: order.map(d => d[0]), y: order.map(d => d[1]),
        marker: { color: order.map(d => (d[0] >= 0 ? COLORS.blue : COLORS.red)) },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
      }], { title: '標準化係數（特徵每增加 1 個標準差，目標改變幾個標準差）', margin: { l: 170, r: 20, t: 48, b: 48 }, height: Math.max(360, 22 * order.length + 100) });

      const n = M.n;
      plot('plotResid', [{
        type: 'scatter', mode: 'markers', x: thin(M.fitted), y: thin(M.resid),
        marker: { color: COLORS.blue, size: 4, opacity: 0.5 }, hovertemplate: '預測 %{x:.3f}<br>殘差 %{y:.3f}<extra></extra>',
      }], {
        title: '殘差 vs 預測值（線性與同質變異）', xaxis: { title: '預測值 (Fitted)' }, yaxis: { title: '殘差 (Residual)' },
        shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: COLORS.red, dash: 'dash' } }],
      });

      const qq = S.qqPoints(M.resid);
      const sd = S.std(M.resid), mu = S.mean(M.resid);
      const t0 = qq.theoretical[0], t1 = qq.theoretical[n - 1];
      plot('plotQQ', [{
        type: 'scatter', mode: 'markers', x: thin(qq.theoretical), y: thin(qq.sample), name: '殘差',
        marker: { color: COLORS.blue, size: 4, opacity: 0.6 },
      }, {
        type: 'scatter', mode: 'lines', x: [t0, t1], y: [mu + sd * t0, mu + sd * t1], name: '常態參考線', line: { color: COLORS.red, dash: 'dash' },
      }], { title: '常態性：Q-Q Plot', xaxis: { title: '理論分位數' }, yaxis: { title: '殘差分位數' }, showlegend: false });

      plot('plotResidHist', [{ type: 'histogram', x: Array.from(M.resid), nbinsx: 50, marker: { color: COLORS.blue, line: { color: COLORS.paper, width: 1 } } }],
        { title: '殘差分佈直方圖', xaxis: { title: '殘差' }, yaxis: { title: '次數' }, bargap: 0.02 });
    },

    eval() {
      const R = state.R, m = R.metrics;
      const cmpRows = R.comparison.map(c => (c.error
        ? [METHOD_LABEL[c.method], c.n, `<span class="hint">${esc(c.error)}</span>`, '', '', '', '', '']
        : [`${METHOD_LABEL[c.method]}${c.method === R.method ? ' <span class="badge good">目前</span>' : ''}`, c.n, fmt(c.trainR2), fmt(c.r2), fmt(c.adjR2), fmt(c.rmse), fmt(c.mae), fmt(c.aic, 1)]));
      const rankByR = R.selNames.map((nm, i) => [nm, Math.abs(S.pearson(R.Xtr.map(r => r[R.sel[i]]), R.ytr))]).sort((a, b) => b[1] - a[1]);

      $('tab-eval').innerHTML = `
        <h2>階段五：Evaluation（模型評估與區間預測圖）</h2>
        <div class="metrics">
          ${metric('R² (測試)', fmt(m.r2))}
          ${metric('Adj. R² (測試)', fmt(m.adjR2))}
          ${metric('RMSE', fmt(m.rmse))}
          ${metric('MAE', fmt(m.mae))}
          ${metric('MSE', fmt(m.mse))}
          ${metric(`${confLabel()} PI 涵蓋率`, `${(R.coverage * 100).toFixed(1)}%`, `測試集落在預測區間內的比例`)}
        </div>
        <div class="card">
          <h3 style="margin-top:0">預測值 vs 真實值（含 ${confLabel()} 信賴區間 CI 與預測區間 PI）</h3>
          <label class="hint">排序依據
            <select id="sortKey"><option value="pred">預測值</option><option value="actual">真實值</option></select>
          </label>
          <div id="plotInterval" class="plot" style="min-height:440px"></div>
          <div class="callout">
            <strong>信賴區間 (CI，深色帶)</strong>：估計「在該特徵組合下，目標的<em>平均值</em>」的範圍，只反映係數估計的不確定性，故較窄。<br>
            <strong>預測區間 (PI，淺色帶)</strong>：估計「<em>單一新樣本</em>」的目標值範圍，額外納入個體誤差 ε 的變異（σ̂²），故較寬。
            測試集實際落在 PI 內的比例為 <strong>${(R.coverage * 100).toFixed(1)}%</strong>，落在 CI 內僅 ${(R.ciCoverage * 100).toFixed(1)}%。
          </div>
        </div>
        <div class="grid-2">
          <div class="card"><div id="plotAvP" class="plot"></div></div>
          <div class="card">
            <h3 style="margin-top:0">單一特徵簡單線性回歸（含 CI / PI 帶）</h3>
            <label class="hint">特徵 <select id="slrFeature">${rankByR.map(([nm]) => `<option>${esc(nm)}</option>`).join('')}</select></label>
            <div id="plotSLR" class="plot"></div>
          </div>
        </div>
        <div class="card">
          <h3 style="margin-top:0">特徵選擇策略比較（同一切分，測試集評估）</h3>
          ${table(['方法', '特徵數', 'R² 訓練', 'R² 測試', 'Adj. R² 測試', 'RMSE', 'MAE', 'AIC'], cmpRows, { rowClass: i => (R.comparison[i].method === R.method ? 'featured' : '') })}
          <p class="hint">若精簡模型的測試 R² 與全特徵相近、但 AIC 更低或 VIF 更小，代表特徵選擇在不犧牲預測力的前提下提升了可解釋性。</p>
        </div>`;

      const drawInterval = () => {
        const key = $('sortKey').value;
        const idx = R.preds.map((_, i) => i).sort((a, b) => (key === 'pred' ? R.preds[a].mean - R.preds[b].mean : R.yte[a] - R.yte[b]));
        const nPts = Math.min(120, idx.length);
        const sub = Array.from({ length: nPts }, (_, i) => idx[Math.round(i * (idx.length - 1) / Math.max(nPts - 1, 1))]);
        const x = sub.map((_, i) => i + 1);
        const g = f => sub.map(i => f(i));
        plot('plotInterval', [
          { x, y: g(i => R.preds[i].piLower), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
          { x, y: g(i => R.preds[i].piUpper), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: COLORS.blueWide, name: `${confLabel()} 預測區間 PI`, hoverinfo: 'skip' },
          { x, y: g(i => R.preds[i].ciLower), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
          { x, y: g(i => R.preds[i].ciUpper), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: COLORS.blueBand, name: `${confLabel()} 信賴區間 CI`, hoverinfo: 'skip' },
          { x, y: g(i => R.preds[i].mean), mode: 'lines', line: { color: COLORS.blue, width: 2 }, name: 'OLS 預測值', hovertemplate: '預測 %{y:.3f}<extra></extra>' },
          { x, y: g(i => R.yte[i]), mode: 'markers', marker: { color: COLORS.ink, size: 6 }, name: '真實值', hovertemplate: '真實 %{y:.3f}<extra></extra>' },
        ], {
          title: `測試集等距抽樣 ${nPts} / ${idx.length} 筆（依${key === 'pred' ? '預測值' : '真實值'}排序）`,
          xaxis: { title: '測試樣本（排序後）' }, yaxis: { title: state.target }, hovermode: 'x unified',
        });
      };
      $('sortKey').addEventListener('change', drawInterval);
      drawInterval();

      const yp = R.preds.map(r => r.mean);
      const lo = Math.min(minOf(R.yte), minOf(yp)), hi = Math.max(maxOf(R.yte), maxOf(yp));
      plot('plotAvP', [{
        type: 'scatter', mode: 'markers', x: thin(R.yte), y: thin(yp), marker: { color: COLORS.blue, size: 5, opacity: 0.55 }, name: '測試樣本',
        hovertemplate: '真實 %{x:.3f}<br>預測 %{y:.3f}<extra></extra>',
      }, {
        type: 'scatter', mode: 'lines', x: [lo, hi], y: [lo, hi], line: { color: COLORS.red, dash: 'dash' }, name: '完美預測 y = x',
      }], { title: '真實值 vs 預測值（測試集）', xaxis: { title: `真實 ${state.target}` }, yaxis: { title: `預測 ${state.target}` } });

      const drawSLR = () => {
        const nm = $('slrFeature').value;
        const j = R.D.names.indexOf(nm);
        const xs = R.Xtr.map(r => r[j]);
        const mod = S.fitOLS(xs.map(v => [v]), R.ytr, [nm]);
        const xmin = minOf(xs), xmax = maxOf(xs);
        const grid = Array.from({ length: 100 }, (_, i) => xmin + (xmax - xmin) * i / 99);
        const pr = mod.predict(grid.map(v => [v]), R.alpha);
        const sx = thin(xs, 2500), sy = thin(R.ytr, 2500);
        plot('plotSLR', [
          { type: 'scatter', mode: 'markers', x: sx, y: sy, marker: { color: COLORS.ink, size: 4, opacity: 0.3 }, name: '訓練樣本' },
          { x: grid, y: pr.map(r => r.piLower), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
          { x: grid, y: pr.map(r => r.piUpper), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: COLORS.blueWide, name: 'PI' },
          { x: grid, y: pr.map(r => r.ciLower), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
          { x: grid, y: pr.map(r => r.ciUpper), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: COLORS.blueBand, name: 'CI' },
          { x: grid, y: pr.map(r => r.mean), mode: 'lines', line: { color: COLORS.blue, width: 2 }, name: '回歸線' },
        ], { title: `${state.target} ~ ${nm}（R² = ${mod.r2.toFixed(3)}）`, xaxis: { title: nm }, yaxis: { title: state.target } });
      };
      $('slrFeature').addEventListener('change', drawSLR);
      drawSLR();
    },

    deploy() {
      const R = state.R;
      const sliders = R.sel.map((j, i) => {
        const col = R.D.X.map(r => r[j]);
        const min = minOf(col), max = maxOf(col);
        const isInt = col.every(v => Number.isInteger(v));
        const step = isInt ? 1 : (max - min) / 200;
        const start = isInt ? Math.round(S.mean(R.Xtr.map(r => r[j]))) : S.mean(R.Xtr.map(r => r[j]));
        return `<div><label for="sim${i}">${esc(R.selNames[i])}：<output id="simOut${i}">${fmt(start, 3)}</output></label>
          <input id="sim${i}" type="range" min="${min}" max="${max}" step="${step}" value="${start}" data-i="${i}"></div>`;
      }).join('');

      $('tab-deploy').innerHTML = `
        <h2>階段六：Deployment（模型部署與即時預測）</h2>
        <p>本網站以純靜態網頁部署於 GitHub Pages：資料解析、特徵選擇與 OLS 估計皆在瀏覽器中以 JavaScript 完成，無需後端伺服器。
        拖動下方 ${R.sel.length} 個入選特徵，即時計算點預測與 ${confLabel()} 區間（滑桿初始值為訓練集平均）。</p>
        <div class="metrics" id="simResult"></div>
        <div class="card"><div id="plotSim" class="plot" style="min-height:220px"></div></div>
        <div class="card"><div class="sliders">${sliders}</div></div>`;

      const update = () => {
        const x = R.sel.map((_, i) => +$(`sim${i}`).value);
        R.sel.forEach((_, i) => { $(`simOut${i}`).textContent = fmt(x[i], 3); });
        const r = R.model.predict([x], R.alpha)[0];
        $('simResult').innerHTML = `
          ${metric(`預測 ${esc(state.target)}`, fmt(r.mean, 3))}
          ${metric(`${confLabel()} 信賴區間 (平均)`, `[${fmt(r.ciLower, 2)}, ${fmt(r.ciUpper, 2)}]`)}
          ${metric(`${confLabel()} 預測區間 (個體)`, `[${fmt(r.piLower, 2)}, ${fmt(r.piUpper, 2)}]`)}`;
        plot('plotSim', [
          { type: 'scatter', mode: 'lines', x: [r.piLower, r.piUpper], y: [0, 0], line: { color: COLORS.blueWide, width: 26 }, name: 'PI' },
          { type: 'scatter', mode: 'lines', x: [r.ciLower, r.ciUpper], y: [0, 0], line: { color: COLORS.blueBand, width: 26 }, name: 'CI' },
          { type: 'scatter', mode: 'markers', x: [r.mean], y: [0], marker: { color: COLORS.ink, size: 14, symbol: 'diamond' }, name: '點預測' },
        ], {
          title: '預測值與區間', height: 260, xaxis: { title: state.target },
          yaxis: { visible: false, range: [-1, 1] }, margin: { l: 30, r: 30, t: 40, b: 96 },
        });
      };
      $('tab-deploy').querySelectorAll('input[type=range]').forEach(el => el.addEventListener('input', update));
      update();
    },

    ai() {
      $('tab-ai').innerHTML = `
        <h2>AI 協助說明與 NotebookLM 研究摘要</h2>
        <div class="card">
          <h3 style="margin-top:0">GPT 輔助內容</h3>
          <div class="callout ai">
            <p style="margin-top:0"><strong>【GPT 輔助內容】</strong>以下部分在 AI 協助下完成：</p>
            <ul>
              <li>依 CRISP-DM 六階段規劃分析架構與網站分頁。</li>
              <li>挑選符合「特徵數 &gt; 13」的 Kaggle 回歸資料集（WHO Life Expectancy、King County House Sales）。</li>
              <li>特徵選擇方法比較（SelectKBest、RFE、LassoCV）與 VIF 共線性診斷的程式設計。</li>
              <li>OLS 信賴區間 / 預測區間公式：CI = ŷ ± t·√(x₀ᵀ Σ̂ x₀)，PI = ŷ ± t·√(x₀ᵀ Σ̂ x₀ + σ̂²)。</li>
              <li>將 Python (statsmodels / scikit-learn) 分析改寫為瀏覽器端 JavaScript，並以 statsmodels 結果交叉驗證數值一致。</li>
            </ul>
            <p style="margin-bottom:0">完整對話紀錄（PDF）：<a href="docs/HW2-chat-history.pdf" target="_blank" rel="noopener">HW2-chat-history.pdf</a>（純文字版：<a href="docs/HW2-chat-history.txt" target="_blank" rel="noopener">txt</a>）。</p>
          </div>
        </div>
        <div class="card">
          <h3 style="margin-top:0">NotebookLM 摘要</h3>
          <div class="callout nlm">
            <p style="margin-top:0"><strong>【NotebookLM 摘要】</strong>主題：WHO 全球預期壽命資料集之多元回歸網路解法研究</p>
            <p style="margin-bottom:0">針對WHO全球預期壽命資料集的多元線性迴歸分析，前處理在缺失值上存在直接刪除、中位數填補或插值法之差異，並常搭配獨熱編碼與對數轉換。對嬰兒與五歲以下死亡率、收入與教育等高度相關特徵，研究多採VIF檢驗或白化處理化解共線性。特徵選擇常用Lasso、逐步迴歸與相關篩選以精簡模型。模型顯示收入指數與疫苗覆蓋率呈顯著正相關（係數為正），成人死亡率與HIV/AIDS呈顯著負相關（係數為負）。線性模型 R² 介於0.77至0.89，RMSE為1.89至4.27歲，但常違反殘差常態性與同變異數假設。總結而言，網路解法雖證實經濟與醫療指標具預測力，但純線性模型受限於假設未滿足與非線性捕捉能力，且缺乏武裝衝突等外部結構變數。</p>
          </div>
        </div>`;
    },
  };

  // ---------- 事件 ----------
  function switchTab(name) {
    state.activeTab = name;
    document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === name));
    document.querySelectorAll('.tab').forEach(s => { s.hidden = s.id !== `tab-${name}`; });
    renderTab(name);
    document.querySelectorAll(`#tab-${name} .js-plotly-plot`).forEach(el => Plotly.Plots.resize(el));
  }

  let timer = null;
  const recomputeSoon = () => { clearTimeout(timer); timer = setTimeout(recompute, 150); };

  document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));

  $('problem').addEventListener('change', e => {
    state.problem = e.target.value;
    $('customBox').hidden = state.problem !== 'custom';
    if (state.problem === 'custom') loadCustomUrl();
    else loadProblem(state.problem).catch(err => setStatus(`資料載入失敗：${err.message}`, true));
  });
  $('loadUrl').addEventListener('click', loadCustomUrl);
  $('csvFile').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    file.text().then(text => useCustomTable(parseCSV(text), `本機上傳：${file.name}`))
      .catch(err => setStatus(`檔案解析失敗：${err.message}`, true));
  });
  $('target').addEventListener('change', e => {
    state.target = e.target.value;
    state.ignore = state.ignore.filter(c => c !== state.target);
    renderCustomControls();
    recompute();
  });
  $('ignoreList').addEventListener('change', () => {
    state.ignore = [...$('ignoreList').querySelectorAll('input:checked')].map(i => i.value);
    recompute();
  });
  $('fsMethod').addEventListener('change', recompute);
  $('kFeatures').addEventListener('input', e => {
    state.k = +e.target.value;
    $('kOut').textContent = state.k;
    recomputeSoon();
  });
  $('testSize').addEventListener('input', e => {
    $('testOut').textContent = (+e.target.value).toFixed(2);
    recomputeSoon();
  });
  $('seed').addEventListener('change', recompute);
  $('conf').addEventListener('change', recompute);

  // 等手寫字型載入後再繪圖，Plotly 才能以正確字寬排版（最多等 3 秒）
  Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 3000))])
    .then(() => loadProblem('p1')).catch(err => setStatus(`資料載入失敗：${err.message}（請透過 http 伺服器開啟，而非 file://）`, true));
})();
