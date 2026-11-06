# 🐈‍⬛ Kuro S-Tier Tactical Web UI Style Prompt v2 (EP-Compressed + Anti-Truncation DNA)

可直接貼給任何 Subagent、LocalSleep-MoE (`:18935`) 或外部 LLM 作為 `system` / 前置風格指令：

```markdown
【UI/UX 視覺與架構鐵律：Kuro Dark-Ops Tactical Console Style v2】
請產出單檔零外部框架依賴（Vanilla HTML5 + CSS3 Grid/Flexbox + 原生 JS + HTML5 Canvas 2D）的「戰術級深色遙測主控台 (Tactical Dark Telemetry Console)」，嚴格遵守以下 7 大設計基因：

1. 🎨 色票合約 (CSS Variables — 深空黑底 × 霓虹語義狀態燈)：
   :root { --bg:#0b0e17; --panel:#111625; --panel-alt:#0d1220; --input-bg:#090d16; --border:#26324d; --text:#e2e8f0; --muted:#8b9bb4; --cyan:#38bdf8; --green:#4ade80; --amber:#f59e0b; --red:#f87171; --purple:#c084fc; }

2. ⚡ 單行緊湊 CSS 鐵律 (Single-Line Compact CSS — 防 Token 截斷護欄)：
   - 所有 CSS Selector 必須寫成「單行緊湊格式」（例如 `.card { background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:12px; }`），嚴禁每個 CSS 屬性單獨換行，將 `<style>` 壓縮在 60 行以內，把 Token 預算留給豐富的 DOM 結構與完整的 `<script>` 渲染邏輯！
   - 嚴禁在 HTML 文字中使用 `$...$` 或 `\rightarrow` 等 LaTeX 語法，一律使用原生 Unicode 符號（`→`、`↔`、`⚡`、`🧬`、`🟢`）。

3. 📐 經典四層戰術拓撲 (4-Tier Full-Viewport Layout)：
   - 全域鎖定 `height:100vh; display:flex; flex-direction:column; overflow:hidden; font-family:"JetBrains Mono","Fira Code","Noto Sans TC",monospace;`
   - [Tier 1 頂部控制艦橋 Header (46px)]：左置 `🐈‍⬛ 系統主標題` + 狀態 `.badge`；右置下拉選單、搜尋框與系統燈號。
   - [Tier 2 盒式遙測數據帶 KPI Ribbon]：採用 4~6 格獨立數據盒 (`.kpi-box { background:var(--panel); border:1px solid var(--border); border-left:3px solid var(--cyan); padding:8px 14px; border-radius:5px; }`)，上方 `.k` 灰字小標，下方 `.v` 18px 粗體高亮數值 + 副標單位。
   - [Tier 3 雙欄非對稱主工作區 Workspace (`display:grid; grid-template-columns: 1fr 400px; flex:1; min-height:0;`)]：
     • 左欄上區：高密度戰術卡片網格 (`.lane-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; padding:14px; overflow-y:auto; }`)。
     • 左欄下區：終端機即時日誌窗 (`.term-box`) + 底部戰術操作按鈕塢 (`.action-bar`)。
     • 右欄：微觀拓撲指標卡 + 底部 HTML5 `<canvas>` 2D 即時矩陣熱力圖/示波器。

4. 🧩 戰術卡片內部解剖學 (3-Layer Rich Card Anatomy — 禁止空洞卡片)：
   每張卡片 (`.lane-card`) 必須同時包含三層資訊：
   - (L1 標頭列)：左側粗體分支/模組名 + 层級徽章 (`[L0]`/`[L1]`/`[L2]`)，右側發光狀態膠囊 (`🟢 CLEAN` / `🟠 DIRTY`)。
   - (L2 微指標雙欄格)：顯示具體技術數據（如 `SHA: 452d022`、`Files: 4 changed`、`Cache: Symlinked`、`Sync: +12 / -2`）。
   - (L3 雙向進度條)：帶數值標籤的 `.bar-bg` 與彩色 `.bar-fill`（依負載自動切換 `--cyan` / `--green` / `--amber` / `--purple`）。

5. 🔍 硬體加速 `<canvas>` 2D 矩陣熱力圖 (Mandatory Immediate Render)：
   - `<canvas>` 必須明確設定 `width="360" height="160"` 屬性，並在 `<script>` 結尾立即呼叫初始化繪圖函式，使用 `ctx.fillRect` 繪製 12×24 雙色熱力格網（正值琥珀橘 `rgba(245,158,11,α)`、負值冰晶藍 `rgba(56,189,248,α)`、零值 `#090d16`）與每 4 格垂直基準線，絕不可留白或全黑！

6. 🛡️ 完賽與自檢要求：
   - 必須直接渲染所有預設卡片與 Canvas 圖形（不可等使用者點擊才渲染），且確保結尾 `</script></body></html>` 100% 閉合無截斷。
```
