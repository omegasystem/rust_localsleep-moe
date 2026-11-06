# 🔧 Teardown 3D Exploded-View Simulator Skill DNA (Adapted for LocalSleep-MoE Slot Forge)
*Distilled from `elvarofficial/teardown` for O(1)-Context Modular Generation + Bundling*

## 🏛️ 核心哲學：為什麼小模型也能寫出 800KB ~ 1MB 級的 3D 拆解模擬器？
連雲端大模型都無法單次吐出 1MB 的單檔 HTML。`teardown` 的秘密在於 **「單一數學真相源 + 4 槽正交分治 + 打包器組裝 (Modular Slot Assembly)」**：

1. **Rule 1 — 單一數學輸入驅動一切 (`One Input Drives Everything`)**：
   所有零件的 3D 座標、旋轉角、爆炸拆解位移 (`explodeFactor: 0.0 ~ 1.0`)，以及畫面上所有儀表板讀數（頻率、功率、溫度、扭矩、高度），**100% 由同一個純數學函式 `computeKinematics(t, params)` 計算得出**，面板數據與 3D 動作永遠同步、絕不脫節。
2. **Rule 2 — 工業白底 × 國際信號橘美學 (`Light Industrial CAD Studio + Signal Orange Accent`)**：
   - 背景採用溫暖工業米白漸層 (`#f4f1ea` $\to$ `#e8e4db`) + 精密極座標工程基準環與十字準星 (`#cbd5e1`)。
   - 機械本體採用消光碳黑 (`#1e2229`)、精密鋁銀 (`#d8dee9`)、深灰結構件 (`#3b4252`)，並在**當前作動/拆解的核心零件**塗上高飽和「國際信號橘 (`#ff4f1f`)」。
   - 四周環繞半透明毛玻璃工程遙測面板 (`rgba(248, 246, 240, 0.88)`，細邊框 `1px solid #d5d0c5`)，搭配等寬工程字體與即時波形圖。
3. **Rule 3 — 每個子系統皆可「爆炸拆解 (Explode)」與「單獨隔離檢視 (Isolate)」**：
   每個 3D 子系統群組 (`THREE.Group`) 必須定義：
   - `homePos: [x, y, z]`（組裝原位）
   - `explodeVec: [dx, dy, dz]`（拆解爆炸方向向量，當 `explodeFactor` 拉高時沿此向量平滑分離，並畫出虛線引線 Leader Lines！）
   - `subsystemId`（對應右側 `SUBSYSTEMS` 面板的 `ATTACHED` / `DETACHED` 橘色切換按鈕）。
4. **Rule 4 — 無人觸碰時自動導覽 (`25s Idle Auto-Tour`)**：
   內建時間軸腳本 (`TOUR_SCRIPT`)：當滑鼠停止操作時自動巡航視角、自動切換爆炸圖與拆解子系統；使用者一旦拖曳或點擊按鈕立即接管控制權。

---

## 🧩 四槽分段鍛造合約 (4-Slot Forge Contract for Local Model)

透過將任務拆為 4 個獨立的小插槽（每個插槽僅 `80 ~ 160` 行 JS，遠低於小模型單次 Context 與偷懶閾值），再由 Python 組裝器嵌入 Three.js 母模：
- **`@@SLOT_1_KINEMATICS@@` (`kin.js`)**：定義真實物理尺寸常數、`SUBSYSTEM_DEFS` 清單（6 大子系統名稱、爆炸向量 `explodeVec`、規格描述），以及純數學函式 `computeKinematics(t, state)` 回傳即時遙測數據與關節角度。
- **`@@SLOT_2_GEOMETRY@@` (`machine.js`)**：使用 `THREE.Group`、`CylinderGeometry`、`SphereGeometry`、`TorusGeometry`、`BoxGeometry` 組合出高細節的 6 大機械子系統，並實作 `updateMachinePose(groups, kin, state)`（處理關節運動、爆炸分離位移 `homePos + explodeVec * explode` 與隔離透明度）。
- **`@@SLOT_3_HUD_PANELS@@` (`console.js`)**：更新左三（主頻率示波器、電力/溫度格、6 軸伺服負載條）、右四（飛行姿態儀、總成快速篩選、顯示模式開關、子系統 `DETACHED/ATTACHED` 拆解清單）、底一（時間軸錄影控制列）的 DOM 與 2D 迷你示波器。
- **`@@SLOT_4_TOUR_DIRECTOR@@` (`live.js`)**：實作視角平滑插值 (`lerp`)、爆炸拆解動畫過渡、以及自動導覽腳本 (`applyTourStep(t, state)`)。
