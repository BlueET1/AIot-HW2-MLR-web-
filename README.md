# 物聯網 HW2: 多元線性回歸分析系統 (Multiple Linear Regression)

🌐 **線上網站**：https://<你的 GitHub 帳號>.github.io/<repo 名稱>/

本專案遵循 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 流程，建置互動式多元線性回歸 (Multiple Linear Regression) 分析與預測系統。

---

## 🌟 專案亮點與作業加分達成情況

* **符合核心作業規範**：
  - 選擇 Kaggle 公開資料集：[Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarrajarshi/life-expectancy-who)
  - 特徵數量：**20 個輸入特徵**（嚴格符合作業 `> 13` 個特徵的限制）。
  - 完整實作**特徵選擇 (Feature Selection)**（包含 Lasso L1 正則化、RFE 遞迴特徵消除、SelectKBest F-檢定）。
  - 模型評估：MSE、RMSE、MAE、$R^2$、Adjusted $R^2$ 與迴歸五大統計假設檢驗。
  - 成果視覺化：**預測圖附帶 95% 信賴區間 (Confidence Interval) 與 95% 預測區間 (Prediction Interval)**。
* **加分項目滿分 (+3 分)**：
  -  **加總分 1 分**：側邊欄提供下拉視窗，支援切換兩種跨領域的經典 Problem：
    1. **Problem 1**：WHO 全球各國預期壽命預測 (20 Features, 公共衛生)
    2. **Problem 2**：美國西雅圖國王郡房價預測 (18 Features → 移除完全共線的 `sqft_basement` 後 17 Features, 房地產經濟)
  -  **加總分 2 分**：支援使用者**任意輸入自訂 Dataset CSV URL**（例如作業示範的 50_Startups）或上傳本機 CSV，具備自動欄位型態識別、類別編碼與回歸建模。

---

## 📊 CRISP-DM 六大分析流程

```mermaid
flowchart TD
    A["1. Business Understanding<br>(業務目標與公衛指標)"] --> B["2. Data Understanding<br>(EDA、相關性熱力圖、分佈)"]
    B --> C["3. Data Preparation<br>(缺失值填補、類別編碼、特徵選擇)"]
    C --> D["4. Modeling<br>(多元線性回歸 OLS、正則化)"]
    D --> E["5. Evaluation<br>(MSE/R²、五大假設、95% CI/PI 區間圖)"]
    E --> F["6. Deployment<br>(GitHub Pages 靜態網站與即時預測)"]
```

### 1. 商業與業務理解 (Business Understanding)
針對聯合國與世界衛生組織 (WHO) 收集的全球 193 國公共衛生指標，探討成人死亡率、疫苗覆蓋率、受教育年數與人均 GDP 等多元因素對國民「人均預期壽命」的邊際效應，協助各國政策制定者量化資源投放優先順序。

### 2. 資料理解 (Data Understanding)
* 樣本數：2,938 筆（涵蓋 2000 ～ 2015 年）。
* 特徵數：20 個輸入特徵，包含健康、死亡率、疫苗覆蓋率、經濟指標。
* 相關性發現：
  - **顯著正相關**：`Schooling` (0.75)、`Income composition of resources` (0.72)、`BMI` (0.57)。
  - **顯著負相關**：`Adult Mortality` (-0.70)、`HIV/AIDS` (-0.56)、`thinness` (-0.48)。

### 3. 資料前處理與特徵選擇 (Data Preparation & Feature Selection)
* 針對數值欄位以中位數填補缺失值，類別欄位 (`Status`) 進行二元虛擬變數編碼。
* 診斷發現 `infant deaths` 與 `under-five deaths`、`thinness 1-19` 與 `thinness 5-9` 存在嚴重共線性（$r > 0.9$）。
* 比較 **Lasso (L1)** 與 **RFE (遞迴特徵消除)**，成功篩選出關鍵 14~17 個獨立特徵，顯著降低 VIF。

### 4. 建模與假設檢定 (Modeling & Diagnostics)
* 使用 `statsmodels.api.OLS` 進行多元線性回歸。
* 五大迴歸統計假設檢定：
  1. **常態性 (Normality)**：繪製 Q-Q Plot 檢驗殘差分佈。
  2. **線性與同質變異 (Homoscedasticity)**：繪製 Fitted vs Residuals 殘差散佈圖。
  3. **獨立性 (Independence)**：計算 Durbin-Watson 統計量（$\approx 2.0$ 代表無自相關）。
  4. **共線性 (Multicollinearity)**：動態 VIF 診斷表。

### 5. 模型評估與區間圖 (Evaluation)
* 測試集績效：
  - **$R^2$**：約 **0.8196**（模型解釋力達 82%）
  - **RMSE**：**3.95 年**
  - **MAE**：**2.93 年**
* **預測區間圖**：透過 OLS 的協方差矩陣精確計算每一筆測試樣本的：
  - **95% 信賴區間 (Mean CI)**：估計母體期望值範圍。
  - **95% 預測區間 (Obs PI)**：涵蓋個體隨機誤差 $\epsilon$ 的單一數值預測區間。

### 6. 模型部署 (Deployment)
網站為純靜態 HTML + JavaScript，部署於 **GitHub Pages**，無需後端伺服器：
* `assets/stats.js`：以 JavaScript 實作 OLS（係數、標準誤、t/p 值、F、AIC/BIC）、95% CI / PI、VIF、SelectKBest、RFE、LassoCV、Jarque–Bera、Breusch–Pagan、Durbin–Watson。數值已與 `statsmodels` / `scikit-learn` 交叉驗證一致（係數、標準誤、CI/PI 相同至 10 位有效數字以上）。
* `assets/app.js`：CRISP-DM 七個分頁、資料集下拉選單、自訂 URL / 上傳、Plotly 圖表與即時預測滑桿。
* 使用者可即時拉動各項特徵拉桿，系統動態給出點預測值與 CI/PI 上下限。

---

## 🤖 AI 協助與 NotebookLM 研究摘要

### 1. NotebookLM 網路解法研究摘要（100 字以上）
> **摘要主題**：基於世界衛生組織 (WHO) 全球預期壽命資料集的多元回歸分析文獻與網路解法探討
> 
> **NotebookLM 研析內容**：
> 根據 NotebookLM 彙整 Kaggle 高分 Notebook 與相關公共衛生回歸研究文獻，針對 WHO Life Expectancy 資料集的分析普遍指出，受教育年數（Schooling）與資源收入指數（Income Composition of Resources）對預期壽命具有最顯著的正向促進效應，而成人死亡率（Adult Mortality）與愛滋病毒普及率（HIV/AIDS）則呈現劇烈負相關。在建模挑戰上，文獻普遍發現幼兒與嬰兒死亡數、不同年齡層消瘦指數間存在高度共線性（VIF > 10），因此必須引入特徵選擇（如 Lasso 正則化或 RFE）以避免模型係數不穩定。相較於單純線性回歸，結合信賴區間與預測區間的診斷方式能有效量化開發中國家在公共衛生政策介入後的不確定性，為決策者提供更可靠的政策調整邊界。

### 2. GPT 輔助內容說明
* 使用 GPT 設計符合 CRISP-DM 規範的分析架構，並輔助撰寫特徵選擇比較演算法與假設檢定程式碼。
* 諮詢對話已透過工具導出為 PDF 存檔備查。

---

## 💻 本地執行指南

### 1. 安裝環境與相依套件
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 執行獨立回歸分析腳本
```bash
python3 analysis.py
```
執行後將在終端機輸出各階段評估指標，並在 `output/prediction_interval_plot.png` 生成高解析度預測區間圖。

### 3. 本機預覽網站
網站需透過 HTTP 伺服器開啟（直接雙擊 `index.html` 會因 `file://` 無法讀取 CSV）：
```bash
python3 -m http.server 8000
```
開啟 `http://localhost:8000` 即可。

### 4. 部署到 GitHub Pages
```bash
git init && git add . && git commit -m "HW2 multiple linear regression"
git branch -M main
git remote add origin https://github.com/<帳號>/<repo>.git
git push -u origin main
```
到 GitHub repo → **Settings → Pages → Build and deployment**，Source 選 **Deploy from a branch**，Branch 選 `main` / `/ (root)`，儲存後約 1 分鐘即可於 `https://<帳號>.github.io/<repo>/` 瀏覽。

