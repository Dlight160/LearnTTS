const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");

// ── Icon Helpers ──
const {
  FaWaveSquare, FaMicrophone, FaVolumeUp, FaArrowRight,
  FaChartLine, FaBrain, FaProjectDiagram, FaCheckCircle,
  FaQuestionCircle, FaLightbulb, FaCogs, FaRandom,
  FaExchangeAlt, FaSearch, FaLayerGroup, FaCode,
  FaClock, FaChartBar, FaArrowDown, FaRedo,
} = require("react-icons/fa");

function renderIconSvg(IconComponent, color = "#000000", size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}

async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

// ── Color Palette (Clean Light) ──
const C = {
  bg:        "F5F5F5",  // off-white background
  bgLight:   "FFFFFF",  // white for table headers / highlight cards
  bgCard:    "FDF3E7",  // warm cream card background
  white:     "2D3436",  // dark charcoal for titles
  ice:       "636E72",  // medium gray for body text
  amber:     "F0A500",  // warm amber accent (kept)
  amberDark: "C88700",  // darker amber
  teal:      "3B82F6",  // blue accent for variety
  muted:     "9CA3A8",  // muted gray for secondary text
  darkBg:    "F0F2F4",  // subtle gray for code/flow blocks
  green:     "10B981",  // emerald green for checkmarks
  red:       "EF4444",  // red for issues/problems
};

// ── Helper Functions ──
function freshShadow() {
  return { type: "outer", color: "000000", blur: 10, offset: 2, angle: 135, opacity: 0.08 };
}

function addBottomWaveform(slide) {
  // Waveform bars along the bottom — signal flow visual motif
  const baseY = 5.59;
  for (let i = 0; i < 14; i++) {
    const h = 0.06 + Math.abs(Math.sin(i * 0.65)) * 0.2;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.3 + i * 0.68, y: baseY - h, w: 0.04, h,
      fill: { color: C.amber, transparency: 75 - i * 2 },
    });
  }
}

function addWaveformMotif(slide, top = true) {
  // Thin amber line with a circle "node" — signal flow visual motif
  const y = top ? 0.15 : 5.25;
  slide.addShape("line", {
    x: 0.4, y: y, w: 9.2, h: 0,
    line: { color: C.amber, width: 1.2, dashType: "solid" },
  });
  // Small circle at the left end
  slide.addShape(pres.shapes.OVAL, {
    x: 0.35, y: y - 0.06, w: 0.12, h: 0.12,
    fill: { color: C.amber },
  });
}

function slideTitle(slide, title, subtitle) {
  addWaveformMotif(slide, true);
  addBottomWaveform(slide);
  slide.addText(title, {
    x: 0.6, y: 0.35, w: 8.8, h: 0.65,
    fontSize: 28, fontFace: "Georgia", color: C.white, bold: true,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6, y: 0.95, w: 8.8, h: 0.4,
      fontSize: 14, fontFace: "Calibri", color: C.muted, italic: true,
      margin: 0,
    });
  }
}

function contentCard(slide, x, y, w, h, opts = {}) {
  return slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.bgCard },
    shadow: opts.noShadow ? undefined : freshShadow(),
  });
}

function bulletText(slide, x, y, w, h, items, opts = {}) {
  const textItems = items.map((item, i) => {
    const isLast = i === items.length - 1;
    if (typeof item === "string") {
      return { text: item, options: { bullet: true, breakLine: !isLast, color: opts.color || C.ice, fontSize: opts.fontSize || 14, fontFace: "Calibri" } };
    }
    // item is an object with text + options
    return {
      text: item.text,
      options: {
        bullet: true,
        breakLine: !isLast,
        color: opts.color || C.ice,
        fontSize: opts.fontSize || 14,
        fontFace: "Calibri",
        ...item.options,
      },
    };
  });
  slide.addText(textItems, { x, y, w, h, valign: "top", margin: [6, 6, 6, 12] });
}

function bigNumber(slide, x, y, number, label, color) {
  slide.addText(number, {
    x, y, w: 2.5, h: 0.7,
    fontSize: 48, fontFace: "Georgia", color: color || C.amber, bold: true,
    align: "center", margin: 0,
  });
  if (label) {
    slide.addText(label, {
      x, y: y + 0.65, w: 2.5, h: 0.4,
      fontSize: 12, fontFace: "Calibri", color: C.muted,
      align: "center", margin: 0,
    });
  }
}

function flowArrow(slide, x, y, w) {
  slide.addText("→", {
    x, y, w, h: 0.4,
    fontSize: 22, fontFace: "Calibri", color: C.amber, bold: true,
    align: "center", valign: "middle", margin: 0,
  });
}

function flowBox(slide, x, y, w, h, text, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.bgLight },
    shadow: freshShadow(),
  });
  slide.addText(text, {
    x, y, w, h,
    fontSize: opts.fontSize || 12, fontFace: "Calibri",
    color: opts.color || C.white, align: "center", valign: "middle",
    margin: 4,
  });
}

function accentBar(slide, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: w || 0.06, h,
    fill: { color: C.amber },
  });
}

// ── Initialize Presentation ──
let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "LearnTTS";
pres.title = "从 Mel 频谱到 VITS——TTS 学习之路";

// ================================================================
// SLIDE 1: 封面
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Decorative waveform lines at bottom
  for (let i = 0; i < 12; i++) {
    const h = 0.08 + Math.abs(Math.sin(i * 0.7)) * 0.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5 + i * 0.78, y: 4.85 - h, w: 0.06, h,
      fill: { color: C.amber, transparency: 70 - i * 3 },
    });
  }

  // Left amber accent line
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 1.5, w: 0.08, h: 2.6,
    fill: { color: C.amber },
  });

  s.addText("从 Mel 频谱到 VITS", {
    x: 0.6, y: 1.5, w: 8.8, h: 1.0,
    fontSize: 42, fontFace: "Georgia", color: C.white, bold: true,
    margin: 0,
  });
  s.addText("TTS 学习之路", {
    x: 0.6, y: 2.5, w: 8.8, h: 0.7,
    fontSize: 30, fontFace: "Georgia", color: C.amber, bold: false,
    margin: 0,
  });
  s.addText("30 分钟技术分享", {
    x: 0.6, y: 3.4, w: 8.8, h: 0.5,
    fontSize: 16, fontFace: "Calibri", color: C.muted,
    margin: 0,
  });
})();

// ================================================================
// SLIDE 2: 全局路线图
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "全局路线图", "三个阶段的接力");

  const stages = [
    { label: "阶段1", title: "语音信号 → Mel 频谱", sub: "语音的数字表示", y: 1.7 },
    { label: "阶段2", title: "Mel 频谱 → 波形", sub: "HiFi-GAN 神经声码器", y: 2.8 },
    { label: "阶段3", title: "文本 → 语音", sub: "VITS 端到端 TTS", y: 3.9 },
  ];

  stages.forEach((st) => {
    accentBar(s, 1.2, st.y, 0.06, 0.65);
    s.addText(st.label, {
      x: 1.5, y: st.y, w: 0.9, h: 0.65,
      fontSize: 11, fontFace: "Calibri", color: C.amber, bold: true, valign: "middle", margin: 0,
    });
    s.addText(st.title, {
      x: 2.5, y: st.y, w: 5.5, h: 0.35,
      fontSize: 18, fontFace: "Georgia", color: C.white, valign: "bottom", margin: 0,
    });
    s.addText(st.sub, {
      x: 2.5, y: st.y + 0.35, w: 5.5, h: 0.3,
      fontSize: 12, fontFace: "Calibri", color: C.muted, valign: "top", margin: 0,
    });
  });

  // Down arrows between stages
  [2.35, 3.45].forEach(y => {
    s.addText("↓", {
      x: 1.5, y, w: 0.6, h: 0.35,
      fontSize: 18, fontFace: "Calibri", color: C.amber, align: "center", valign: "middle", margin: 0,
    });
  });
})();

// ================================================================
// SLIDE 3: 阶段1 概览
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "阶段1：Mel 频谱", "语音的数字表示");

  const topics = [
    { num: "01", title: "波形 vs 频谱", desc: "同一个声音的两种视角" },
    { num: "02", title: "为什么用 Mel", desc: "三种表示的对比 —— 为什么是 80 维" },
    { num: "03", title: "Mel 怎么做", desc: "切帧 → 加窗 → FFT → Mel 滤波器组" },
    { num: "04", title: "Mel 的局限", desc: "Griffin-Lam 重建 —— 引出阶段2" },
  ];

  topics.forEach((t, i) => {
    const y = 1.7 + i * 0.85;
    accentBar(s, 1.0, y, 0.06, 0.6);
    s.addText(t.num, {
      x: 1.3, y, w: 0.5, h: 0.6,
      fontSize: 18, fontFace: "Georgia", color: C.amber, bold: true, valign: "middle", margin: 0,
    });
    s.addText(t.title, {
      x: 1.9, y: y + 0.02, w: 3.0, h: 0.35,
      fontSize: 17, fontFace: "Georgia", color: C.white, valign: "bottom", margin: 0,
    });
    s.addText(t.desc, {
      x: 1.9, y: y + 0.35, w: 6.5, h: 0.25,
      fontSize: 12, fontFace: "Calibri", color: C.muted, valign: "top", margin: 0,
    });
  });
})();

// ================================================================
// SLIDE 4: 波形 vs 频谱
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "波形 vs 频谱", "阶段1：同一个声音的两种视角");

  // Waveform card
  contentCard(s, 0.5, 1.6, 4.3, 3.2);
  accentBar(s, 0.5, 1.6, 0.06, 3.2);
  s.addText("波形 Waveform", {
    x: 0.8, y: 1.75, w: 3.8, h: 0.45,
    fontSize: 18, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });
  bulletText(s, 0.8, 2.3, 3.8, 2.3, [
    "横轴 = 时间，纵轴 = 振幅",
    "文件里每个浮点数 = 声波振到什么位置",
    "正数 → 振膜向外推，负数 → 向里拉",
    "每秒 16000 个点 = 16kHz 采样率",
    { text: "看不出「男声还是女声，是 /a/ 还是 /i/」", options: { color: C.red, italic: true } },
  ]);

  // Arrow
  s.addText("→", {
    x: 4.5, y: 2.8, w: 0.8, h: 0.5,
    fontSize: 32, fontFace: "Calibri", color: C.amber, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  // Spectrogram card
  contentCard(s, 5.2, 1.6, 4.3, 3.2);
  accentBar(s, 5.2, 1.6, 0.06, 3.2);
  s.addText("频谱 Spectrogram", {
    x: 5.5, y: 1.75, w: 3.8, h: 0.45,
    fontSize: 18, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });
  bulletText(s, 5.5, 2.3, 3.8, 2.3, [
    "横轴 = 时间，纵轴 = 频率，颜色 = 能量",
    "能量集中在低频 → 男声",
    "高频横杠 → 摩擦音 /s/",
    "一眼看出音色、音高、内容",
    { text: "✓ 语音的本质 = 不同频率的能量随时间变化", options: { color: C.green } },
  ]);
})();

// ================================================================
// SLIDE 4: 为什么用 Mel 频谱
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "为什么用 Mel 频谱？", "三种表示的对比");

  // Three cards
  const cards = [
    { title: "原始波形", items: ["1秒 16000+ 采样点", "对相位过敏感", "人耳对相位不敏感 → 模型浪费容量"], verdict: "维度太高" },
    { title: "线性频谱", items: ["1024 维", "人耳非线性感知", "低频敏感，高频迟钝"], verdict: "不符人耳" },
    { title: "Mel 频谱", items: ["80 维（人耳感知标尺）", "低频 bin 窄，高频 bin 宽", "丢弃人耳不敏感信息"], verdict: "最优平衡", isBest: true },
  ];

  cards.forEach((card, i) => {
    const x = 0.5 + i * 3.15;
    const fill = card.isBest ? C.bgLight : C.bgCard;
    contentCard(s, x, 1.6, 2.95, 3.0, { fill });
    if (card.isBest) {
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: 1.6, w: 2.95, h: 0.04, fill: { color: C.amber },
      });
    }
    s.addText(card.title, {
      x: x + 0.2, y: 1.75, w: 2.55, h: 0.4,
      fontSize: 16, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
    });
    bulletText(s, x + 0.2, 2.25, 2.55, 1.5, card.items, { fontSize: 11 });
    s.addText(card.verdict, {
      x: x + 0.2, y: 3.95, w: 2.55, h: 0.4,
      fontSize: 13, fontFace: "Calibri", color: card.isBest ? C.amber : C.red,
      bold: true, margin: 0,
    });
  });

  // Key stat
  s.addText("80 维 × 86帧/秒，比波形压缩约 256 倍", {
    x: 0.5, y: 4.55, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.amber, align: "center", italic: true, margin: 0,
  });
})();

// ================================================================
// SLIDE 5: Mel 频谱怎么做
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "Mel 频谱怎么做？", "从音频文件到 Mel 频谱的完整链路");

  // Flow diagram
  const steps = [
    { label: "波形", sub: "一维采样序列" },
    { label: "切帧", sub: "帧长 25ms" },
    { label: "加窗", sub: "Hann 窗" },
    { label: "FFT", sub: "线性频谱" },
    { label: "Mel 滤波", sub: "80 维输出" },
  ];

  steps.forEach((st, i) => {
    const x = 0.5 + i * 1.9;
    flowBox(s, x, 2.0, 1.6, 0.85, `${st.label}\n${st.sub}`, { fontSize: 11, fill: i === 4 ? C.bgLight : C.bgCard });
    if (i < steps.length - 1) {
      s.addText("→", {
        x: x + 1.6, y: 2.15, w: 0.3, h: 0.55,
        fontSize: 20, fontFace: "Calibri", color: C.amber, bold: true,
        align: "center", valign: "middle", margin: 0,
      });
    }
  });

  // Key parameters
  s.addText("关键参数", {
    x: 0.5, y: 3.15, w: 9, h: 0.35,
    fontSize: 15, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  const params = [
    ["n_fft", "频率分辨率 vs 时间分辨率的权衡", "越大 → 频率分辨率 ↑，时间分辨率 ↓"],
    ["hop_length", "帧移（相邻帧重叠 ~60%）", "越小 → 时间分辨率 ↑，数据量 ↑"],
    ["n_mels = 80", "Mel 尺度上等距的三角滤波器", "1024 维 → 80 维，降维 + 感知校准"],
  ];

  params.forEach((p, i) => {
    const y = 3.55 + i * 0.45;
    s.addText(p[0], {
      x: 0.7, y, w: 2.0, h: 0.35,
      fontSize: 13, fontFace: "Consolas", color: C.amber, bold: true, margin: 0,
    });
    s.addText(p[1], {
      x: 2.8, y, w: 3.2, h: 0.35,
      fontSize: 12, fontFace: "Calibri", color: C.ice, margin: 0,
    });
    s.addText(p[2], {
      x: 6.1, y, w: 3.5, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: C.muted, italic: true, margin: 0,
    });
  });
})();

// ================================================================
// SLIDE 6: Mel 的局限
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "Mel 频谱的局限", "Griffin-Lam 重建 → 引出阶段2");

  contentCard(s, 0.5, 1.6, 9.0, 3.2);

  s.addText("Griffin-Lam 从 Mel 重建波形", {
    x: 0.8, y: 1.75, w: 8.4, h: 0.45,
    fontSize: 20, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  bulletText(s, 0.8, 2.35, 8.4, 1.8, [
    "STFT 输出是复数（幅度 + 相位），Mel 只保留幅度，丢弃相位",
    "GL 算法从幅度迭代猜测相位 → 重建音频",
    "听感对比：原始 vs GL 重建 → 明显「纸杯声」",
    { text: "Mel 是高效的中间表示，但从它重建回波形 — 质量有限", options: { bold: true, color: C.white } },
  ], { fontSize: 14 });

  s.addText("🎤  播放 GL 重建 vs 原始音频对比", {
    x: 0.8, y: 4.25, w: 8.4, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.amber, italic: true, margin: 0,
  });
})();

// ================================================================
// SLIDE 7: 阶段2 概览
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "阶段2：HiFi-GAN", "神经声码器 · Mel → 波形");

  const topics = [
    { num: "01", title: "核心问题", desc: "Mel(80,T) → 波形(1,T×256)，256倍上采样" },
    { num: "02", title: "Generator 逐级上采样", desc: "×8 → ×8 → ×2 → ×2，每级贡献不同" },
    { num: "03", title: "MRF 三路抛光", desc: "不同感受野同时检查，不留死角" },
    { num: "04", title: "三种 Loss 的分工", desc: "Mel Loss（内容）+ GAN Loss（逼真）+ FM Loss（稳定）" },
    { num: "05", title: "MPD + MSD 双判别器", desc: "周期正确 + 局部波形干净，缺一不可" },
  ];

  topics.forEach((t, i) => {
    const y = 1.5 + i * 0.72;
    accentBar(s, 1.0, y, 0.06, 0.55);
    s.addText(t.num, {
      x: 1.3, y, w: 0.5, h: 0.55,
      fontSize: 18, fontFace: "Georgia", color: C.amber, bold: true, valign: "middle", margin: 0,
    });
    s.addText(t.title, {
      x: 1.9, y: y + 0.02, w: 3.0, h: 0.3,
      fontSize: 17, fontFace: "Georgia", color: C.white, valign: "bottom", margin: 0,
    });
    s.addText(t.desc, {
      x: 1.9, y: y + 0.3, w: 6.5, h: 0.25,
      fontSize: 12, fontFace: "Calibri", color: C.muted, valign: "top", margin: 0,
    });
  });
})();

// ================================================================
// SLIDE 7: HiFi-GAN 核心问题
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "HiFi-GAN：从 Mel 到波形的超分辨率", "阶段2 · 核心问题");

  // Big number
  bigNumber(s, 0.7, 1.8, "256×", "上采样倍率", C.amber);

  s.addText("Mel (80, T)  →  波形 (1, T×256)", {
    x: 3.5, y: 1.9, w: 6, h: 0.6,
    fontSize: 24, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  contentCard(s, 0.5, 2.9, 9.0, 2.3);
  bulletText(s, 0.8, 3.1, 8.4, 1.9, [
    "这不是简单的反变换——这是一个生成问题",
    "GL 重建像「蒙着布说话」→ 需要学一个解码器",
    { text: "核心思路：用 GAN 对抗训练，让判别器逼 Generator 生成逼真波形", options: { bold: true, color: C.white } },
    "逐级上采样 ×8 → ×8 → ×2 → ×2，每级用 MRF 抛光",
  ], { fontSize: 14 });
})();

// ================================================================
// SLIDE 8: Generator 渐进式上采样
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "Generator：逐级上采样，每级贡献不同", "×8 → ×8 → ×2 → ×2");

  const levels = [
    { label: "up1 + MRF1", scale: "×8", role: "节奏和响度", detail: "从帧级信息展开到音节级", metaphor: "头在哪儿、身子在哪儿", x: 0.4 },
    { label: "up2 + MRF2", scale: "×8", role: "旋律轮廓", detail: "共振峰过渡 + 基频连续变化", metaphor: "五官位置——表情", x: 2.65 },
    { label: "up3 + MRF3", scale: "×2", role: "高频谐波纹理", detail: "摩擦音噪声、气息声", metaphor: "皮肤纹理", x: 4.9 },
    { label: "up4 + MRF4", scale: "×2", role: "气息/摩擦音细节", detail: "最后抛光 → 真人质感", metaphor: "就是真人", x: 7.15 },
  ];

  levels.forEach(lv => {
    contentCard(s, lv.x, 1.55, 2.1, 2.8);
    s.addShape(pres.shapes.RECTANGLE, {
      x: lv.x, y: 1.55, w: 2.1, h: 0.04, fill: { color: C.amber },
    });
    s.addText(lv.label, {
      x: lv.x + 0.1, y: 1.7, w: 1.9, h: 0.3,
      fontSize: 13, fontFace: "Consolas", color: C.amber, margin: 0,
    });
    s.addText(lv.scale, {
      x: lv.x + 0.1, y: 1.95, w: 1.9, h: 0.4,
      fontSize: 28, fontFace: "Georgia", color: C.white, bold: true, margin: 0, align: "center",
    });
    s.addText(lv.role, {
      x: lv.x + 0.1, y: 2.45, w: 1.9, h: 0.3,
      fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, margin: 0, align: "center",
    });
    s.addText(lv.detail, {
      x: lv.x + 0.1, y: 2.75, w: 1.9, h: 0.5,
      fontSize: 10, fontFace: "Calibri", color: C.muted, margin: 0, align: "center",
    });
    s.addText(`「${lv.metaphor}」`, {
      x: lv.x + 0.1, y: 3.3, w: 1.9, h: 0.35,
      fontSize: 10, fontFace: "Calibri", color: C.amber, italic: true, margin: 0, align: "center",
    });
  });

  // Why 4 steps
  s.addText("为什么分 4 步，不一步到位 ×256？", {
    x: 0.5, y: 4.4, w: 4.5, h: 0.3,
    fontSize: 14, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });
  bulletText(s, 0.5, 4.75, 9.0, 0.3, [
    "一步 ×256 → 棋盘格伪影严重，参数爆炸。分步：每步小幅拉伸(kernel=16/4) + MRF 及时抛光，逐级累积不留死角",
  ], { fontSize: 12 });
})();

// ================================================================
// SLIDE 9: MRF 三路抛光
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "MRF：三路并发的「抛光」机制", "Multi-Receptive Field Fusion");

  const routes = [
    { k: 3, dil: "[1,3,5]", range: "极高频纹理", desc: "摩擦音的噪声" },
    { k: 7, dil: "[1,3,5]", range: "中高频过渡", desc: "共振峰迁移" },
    { k: 11, dil: "[1,3,5]", range: "低频周期结构", desc: "基频 F0" },
  ];

  routes.forEach((r, i) => {
    const x = 0.5 + i * 3.15;
    contentCard(s, x, 1.6, 2.95, 2.2);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.6, w: 2.95, h: 0.04, fill: { color: C.amber },
    });
    s.addText(`ResBlock(k=${r.k}, dil=${r.dil})`, {
      x: x + 0.15, y: 1.75, w: 2.65, h: 0.4,
      fontSize: 12, fontFace: "Consolas", color: C.amber, margin: 0, align: "center",
    });
    s.addText(r.range, {
      x: x + 0.15, y: 2.3, w: 2.65, h: 0.35,
      fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, margin: 0, align: "center",
    });
    s.addText(r.desc, {
      x: x + 0.15, y: 2.7, w: 2.65, h: 0.35,
      fontSize: 13, fontFace: "Calibri", color: C.muted, margin: 0, align: "center",
    });
  });

  // Arrow down to merge
  s.addText("↓  三路求和  ↓", {
    x: 1.5, y: 3.9, w: 7, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.amber, align: "center", valign: "middle", margin: 0,
  });

  s.addText("三把不同粗细的锉刀同时抛光，不留死角", {
    x: 0.5, y: 4.4, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.white, align: "center", italic: true, margin: 0,
  });
})();

// ================================================================
// SLIDE 10: 三种 Loss 分工
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "三种 Loss 的分工", "只用 Mel Loss → 输出「平均化」的模糊波形");

  const headers = ["Loss", "权重", "作用", "比喻"];
  const rows = [
    ["Mel Loss", "×45", "内容保真度，约束频谱包络", "地图 — 内容别错"],
    ["GAN Loss", "×1", "逼真度，骗过判别器", "演技 — 自然不假"],
    ["FM Loss", "×2", "稠密梯度，稳定训练", "辅导 — 细节纠正"],
  ];

  const colW = [1.8, 0.8, 3.2, 3.2];
  const headerRow = headers.map((h, i) => ({
    text: h,
    options: { fill: { color: C.bgLight }, color: C.white, bold: true, fontSize: 14, fontFace: "Calibri", align: "center", valign: "middle" },
  }));

  const dataRows = rows.map(row =>
    row.map((cell, ci) => ({
      text: cell,
      options: {
        fill: { color: C.bgCard }, color: ci === 1 ? C.amber : C.ice,
        bold: ci === 1, fontSize: 13, fontFace: ci === 1 ? "Georgia" : "Calibri",
        align: "center", valign: "middle",
      },
    }))
  );

  s.addTable([headerRow, ...dataRows], {
    x: 0.5, y: 1.6, w: 9.0, colW,
    border: { pt: 1, color: C.bgLight },
    rowH: [0.5, 0.55, 0.55, 0.55],
  });

  s.addText("GAN 让判别器看完整波形（12800 维），发现 Mel 看不到的失真", {
    x: 0.5, y: 3.7, w: 9, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.muted, italic: true, margin: 0,
  });
})();

// ================================================================
// SLIDE 11: MPD + MSD 双判别器
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "MPD + MSD 双判别器架构", "缺一不可：周期正确 + 局部波形干净");

  const cols = [
    {
      title: "MPD · 多周期判别器",
      items: ["按周期重排 1D → 2D", "检查周期结构是否自然", "防止：音高抖动"],
      effect: "只用 MPD → 周期对了但内部波形粗糙",
    },
    {
      title: "MSD · 多尺度判别器",
      items: ["下采样后看不同尺度", "检查整体质感", "防止：包络异常"],
      effect: "只用 MSD → 整体平滑但周期边界模糊",
    },
  ];

  cols.forEach((col, i) => {
    const x = 0.5 + i * 4.7;
    contentCard(s, x, 1.6, 4.4, 2.5);
    accentBar(s, x, 1.6, 0.06, 2.5);
    s.addText(col.title, {
      x: x + 0.25, y: 1.75, w: 3.9, h: 0.4,
      fontSize: 18, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
    });
    bulletText(s, x + 0.25, 2.25, 3.9, 1.2, col.items, { fontSize: 13 });
    s.addText(col.effect, {
      x: x + 0.25, y: 3.6, w: 3.9, h: 0.4,
      fontSize: 12, fontFace: "Calibri", color: C.red, italic: true, margin: 0,
    });
  });

  s.addText("🎤  播放 GL vs HiFi-GAN 对比音频", {
    x: 0.5, y: 4.4, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.amber, italic: true, margin: 0,
  });
})();

// ================================================================
// SLIDE 13: 阶段3 概览
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "阶段3：VITS", "端到端 TTS · 文本 → 语音");

  const topics = [
    { num: "01", title: "Tacotron2 的两大硬伤", desc: "自回归误差累积 + Attention 对齐不稳定" },
    { num: "02", title: "VITS 架构概览", desc: "VAE 框架：用隐变量 z 替代人工设计的 Mel" },
    { num: "03", title: "MAS 无监督对齐", desc: "动态规划找最优单调路径，不可能跳词" },
    { num: "04", title: "SDP 随机时长预测器", desc: "预测时长分布，采样带来语速变化" },
    { num: "05", title: "Normalizing Flow", desc: "把简单高斯「揉捏」成复杂分布" },
    { num: "06", title: "KL Loss", desc: "训练核心推动力 —— 让先验学会后验" },
    { num: "07", title: "VITS 的后续影响", desc: "CosyVoice（Flow Matching） + FishSpeech（LLM）" },
  ];

  topics.forEach((t, i) => {
    const y = 1.4 + i * 0.55;
    accentBar(s, 1.0, y, 0.06, 0.4);
    s.addText(t.num, {
      x: 1.3, y, w: 0.5, h: 0.4,
      fontSize: 16, fontFace: "Georgia", color: C.amber, bold: true, valign: "middle", margin: 0,
    });
    s.addText(t.title, {
      x: 1.9, y: y + 0.02, w: 3.2, h: 0.22,
      fontSize: 15, fontFace: "Georgia", color: C.white, valign: "bottom", margin: 0,
    });
    s.addText(t.desc, {
      x: 1.9, y: y + 0.22, w: 6.5, h: 0.2,
      fontSize: 11, fontFace: "Calibri", color: C.muted, valign: "top", margin: 0,
    });
  });
})();

// ================================================================
// SLIDE 12: Tacotron2 两大硬伤
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "Tacotron2 的两大硬伤", "阶段3 · 为什么需要 VITS");

  const issues = [
    { title: "自回归误差累积", items: ["逐帧预测 Mel，一步错步步错", "容易 babbling 或尾部崩溃", "根本原因：AR 解码无容错机制"] },
    { title: "Soft Attention 对齐不稳定", items: ["可能跳词、重复词、对齐崩溃", "每步动态计算注意力权重", "部署最大的拦路虎"] },
  ];

  issues.forEach((issue, i) => {
    const y = 1.6 + i * 1.7;
    contentCard(s, 0.5, y, 9.0, 1.5);
    accentBar(s, 0.5, y, 0.06, 1.5);
    s.addText(`问题 ${i + 1}：${issue.title}`, {
      x: 0.8, y: y + 0.15, w: 8.4, h: 0.35,
      fontSize: 17, fontFace: "Georgia", color: C.red, bold: true, margin: 0,
    });
    bulletText(s, 0.8, y + 0.55, 8.4, 0.8, issue.items, { fontSize: 12 });
  });

  s.addText("→  VITS 用 MAS + 非自回归架构一次性解决", {
    x: 0.5, y: 4.85, w: 9, h: 0.3,
    fontSize: 15, fontFace: "Calibri", color: C.amber, bold: true, align: "center", margin: 0,
  });
})();

// ================================================================




// SLIDE: VITS 架构概览
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "VITS 架构概览", "VAE 框架 · 端到端 TTS");

  const boxH = 0.28;
  const arrowH = 0.06;
  const gapH = 0.06;
  const fs = 9;
  const boxFill = C.bgCard;     // white boxes on off-white bg
  const ioFill = C.bgLight;      // slightly darker for input/output
  const makeShadow = () => freshShadow();

  function flowBox(x, y, w, h, text, fill) {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, shadow: makeShadow() });
    s.addText(text, { x, y, w, h, fontSize: fs, fontFace: "Calibri", color: C.white, align: "center", valign: "middle", margin: 2 });
  }

  function downArrow(x, y) {
    s.addText("↓", { x, y, w: 0.3, h: arrowH, fontSize: 10, fontFace: "Calibri", color: C.amber, align: "center", valign: "middle", margin: 0 });
    return y + arrowH + gapH;
  }

  // ── Left: Training ──
  s.addText("训练流程（有真实音频）", {
    x: 0.3, y: 1.4, w: 4.7, h: 0.28,
    fontSize: 14, fontFace: "Georgia", color: C.amber, bold: true, margin: 0,
  });

  let y = 1.78;
  const L = 0.3;
  const LW = 4.7;

  flowBox(L, y, 2.15, boxH, "文本", ioFill);
  flowBox(L + 2.35, y, 2.35, boxH, "真实波形", ioFill);
  y += boxH;
  y = downArrow(L + 0.95, y);
  s.addText("↓", { x: L + 3.3, y: y - arrowH - gapH, w: 0.3, h: arrowH, fontSize: 10, fontFace: "Calibri", color: C.amber, align: "center", valign: "middle", margin: 0 });

  flowBox(L, y, 2.15, boxH, "TextEncoder → h_text", boxFill);
  flowBox(L + 2.35, y, 2.35, boxH, "STFT → 线性频谱", boxFill);
  y += boxH;
  y = downArrow(L + 0.95, y);
  s.addText("↓", { x: L + 3.3, y: y - arrowH - gapH, w: 0.3, h: arrowH, fontSize: 10, fontFace: "Calibri", color: C.amber, align: "center", valign: "middle", margin: 0 });

  flowBox(L, y, 2.15, boxH, "MAS 找对齐", boxFill);
  flowBox(L + 2.35, y, 2.35, boxH, "PosteriorEncoder", boxFill);
  y += boxH;
  y = downArrow(L + 0.95, y);
  s.addText("↓", { x: L + 3.3, y: y - arrowH - gapH, w: 0.3, h: arrowH, fontSize: 10, fontFace: "Calibri", color: C.amber, align: "center", valign: "middle", margin: 0 });

  flowBox(L, y, 2.15, boxH, "按时长展开到帧级", boxFill);
  flowBox(L + 2.35, y, 2.35, boxH, "后验分布 q(z|x)", boxFill);
  y += boxH;

  // KL Loss bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: L, y: y + 0.02, w: LW, h: arrowH,
    fill: { color: C.bgCard },
  });
  s.addText("↕  KL Loss（让先验学会后验）", {
    x: L, y: y + 0.02, w: LW, h: arrowH,
    fontSize: 9, fontFace: "Calibri", color: C.amber, bold: true, align: "center", valign: "middle", margin: 0,
  });
  y += arrowH + gapH;

  flowBox(L, y, LW, boxH, "Normalizing Flow（先验增强）", boxFill);
  y += boxH;
  y = downArrow(L + 2.2, y);

  flowBox(L, y, LW, boxH, "先验分布 p(z|c)", boxFill);
  y += boxH;
  y = downArrow(L + 2.2, y);

  flowBox(L, y, LW, boxH, "采样 z（192 维 × T 帧）", boxFill);
  y += boxH;
  y = downArrow(L + 2.2, y);

  flowBox(L, y, LW, boxH, "HiFi-GAN Decoder → 波形 ✓", ioFill);

  // ── Right: Inference ──
  const R = 5.3;
  const RW = 4.4;

  s.addText("推理流程（只有文本）", {
    x: R, y: 1.4, w: RW, h: 0.28,
    fontSize: 14, fontFace: "Georgia", color: C.amber, bold: true, margin: 0,
  });

  let ry = 1.78;

  flowBox(R, ry, RW, boxH, "文本", ioFill);
  ry += boxH;
  ry = downArrow(R + 2.05, ry);

  flowBox(R, ry, RW, boxH, "TextEncoder → h_text（音素级）", boxFill);
  ry += boxH;
  ry = downArrow(R + 2.05, ry);

  flowBox(R, ry, RW, boxH, "SDP 随机时长预测器", boxFill);
  ry += boxH;
  ry = downArrow(R + 2.05, ry);

  flowBox(R, ry, RW, boxH, "按时长展开到帧级", boxFill);
  ry += boxH;
  ry = downArrow(R + 2.05, ry);

  flowBox(R, ry, RW, boxH, "Normalizing Flow（先验增强）", boxFill);
  ry += boxH;
  ry = downArrow(R + 2.05, ry);

  flowBox(R, ry, RW, boxH, "先验分布 p(z|c)", boxFill);
  ry += boxH;
  ry = downArrow(R + 2.05, ry);

  flowBox(R, ry, RW, boxH, "采样 z（192 维 × T 帧）", boxFill);
  ry += boxH;
  ry = downArrow(R + 2.05, ry);

  flowBox(R, ry, RW, boxH, "HiFi-GAN Decoder → 波形 ✓", ioFill);

  // Key differences — placed below both columns
  s.addText("训练 vs 推理的关键区别：MAS 需要真实音频算 z → 推理时用 SDP 预测时长替代；PosteriorEncoder 训练后丢弃", {
    x: 0.5, y: 4.96, w: 9, h: 0.25,
    fontSize: 10, fontFace: "Calibri", color: C.muted, italic: true, margin: 0,
  });
})();
// SLIDE 15: MAS 无监督对齐
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "MAS：无监督自动对齐", "动态规划找最优单调路径");

  // Problem
  s.addText("问题：10 个音素 vs 200 帧 → 谁对应谁？", {
    x: 0.5, y: 1.5, w: 9, h: 0.4,
    fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });

  // Solution steps
  const steps = [
    { title: "1. 构造相似度矩阵", desc: "h_text[音素] 和 z[帧] 的余弦相似度" },
    { title: "2. 动态规划", desc: "DP[i,j] = max(DP[i-1,j-1], DP[i,j-1]) + sim[i,j]" },
    { title: "3. 提取时长", desc: "数每个音素覆盖了多少帧 → 时长序列" },
  ];

  steps.forEach((st, i) => {
    const x = 0.5 + i * 3.15;
    contentCard(s, x, 2.1, 2.95, 1.3);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 2.1, w: 2.95, h: 0.04, fill: { color: C.amber },
    });
    s.addText(st.title, {
      x: x + 0.15, y: 2.25, w: 2.65, h: 0.35,
      fontSize: 14, fontFace: "Georgia", color: C.amber, bold: true, margin: 0,
    });
    s.addText(st.desc, {
      x: x + 0.15, y: 2.65, w: 2.65, h: 0.55,
      fontSize: 12, fontFace: "Calibri", color: C.ice, margin: 0,
    });
  });

  // Constraints
  s.addText("约束：单调 · 连续 · 每个音素至少 1 帧 · 起点终点固定", {
    x: 0.5, y: 3.55, w: 9, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: C.muted, italic: true, margin: 0,
  });

  // Comparison table
  const headers = ["", "Tacotron2 Attention", "VITS MAS"];
  const rows = [
    ["对齐方式", "每步动态计算（soft）", "全局搜索最优（hard）"],
    ["单调性", "Loss 间接约束", "算法内建，数学保证"],
    ["稳定性", "可能跳词/重复/崩溃", "不可能跳词/重复"],
  ];

  const colW = [1.5, 3.75, 3.75];
  const headerRow = headers.map((h, i) => ({
    text: h,
    options: { fill: { color: C.bgLight }, color: C.white, bold: true, fontSize: 12, fontFace: "Calibri", align: "center", valign: "middle" },
  }));
  const dataRows = rows.map((row, ri) =>
    row.map((cell, ci) => ({
      text: cell,
      options: {
        fill: { color: C.bgCard }, color: ci === 2 ? C.green : (ci === 1 && ri === 2 ? C.red : C.ice),
        bold: ci >= 1 && ri === 2, fontSize: 11, fontFace: "Calibri", align: "center", valign: "middle",
      },
    }))
  );

  s.addTable([headerRow, ...dataRows], {
    x: 0.5, y: 4.05, w: 9.0, colW,
    border: { pt: 1, color: C.bgLight },
    rowH: [0.4, 0.35, 0.35, 0.35],
  });
})();

// ================================================================
// SLIDE 16: SDP 随机时长预测器
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "SDP：随机时长预测器", "MAS「能看不能用」→ SDP 来替代");

  // Why needed
  s.addText("为什么需要 SDP？", {
    x: 0.5, y: 1.5, w: 9, h: 0.35,
    fontSize: 16, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });
  bulletText(s, 0.5, 1.9, 9.0, 0.6, [
    "MAS 需要真实音频的 z 才能算对齐 → 推理时没有 → 不能跑",
    "训练时用 MAS 得到「标准答案」时长 d → 推理时需要不看音频就能预测时长的模块",
  ], { fontSize: 13 });

  // Why stochastic
  s.addText("为什么是「随机」的？", {
    x: 0.5, y: 2.7, w: 9, h: 0.35,
    fontSize: 16, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  const modes = [
    { title: "确定性预测", code: "/a/ → 4 帧（每次一样）", desc: "机器人感，缺少自然变化" },
    { title: "SDP 分布预测", code: "/a/ → 𝒩(4, 0.5)", desc: "每次采样不同 → 语速自然变化" },
  ];

  modes.forEach((m, i) => {
    const x = 0.5 + i * 4.7;
    contentCard(s, x, 3.15, 4.4, 1.15);
    accentBar(s, x, 3.15, 0.06, 1.15);
    s.addText(m.title, {
      x: x + 0.25, y: 3.25, w: 3.9, h: 0.35,
      fontSize: 14, fontFace: "Georgia", color: i === 1 ? C.amber : C.white, bold: true, margin: 0,
    });
    s.addText(m.code, {
      x: x + 0.25, y: 3.6, w: 3.9, h: 0.3,
      fontSize: 12, fontFace: "Consolas", color: C.ice, margin: 0,
    });
    s.addText(m.desc, {
      x: x + 0.25, y: 3.9, w: 3.9, h: 0.3,
      fontSize: 11, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });

  // Training vs inference
  s.addText("精妙之处：SDP 学习 MAS 自动发现的时长 — 互相配合但互不依赖", {
    x: 0.5, y: 4.4, w: 9, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.amber, italic: true, margin: 0,
  });

  const trainingInference = [
    ["", "训练", "推理"],
    ["时长来源", "MAS（需真实音频）", "SDP 采样"],
    ["SDP 在干嘛", "学习 — 最大化 MAS 时长的似然", "干活 — 从学到的分布采样"],
    ["Loss", "Negative Log-Likelihood", "无"],
  ];

  const colW2 = [1.8, 3.6, 3.6];
  const headerRow2 = trainingInference[0].map(h => ({
    text: h, options: { fill: { color: C.bgLight }, color: C.white, bold: true, fontSize: 11, fontFace: "Calibri", align: "center", valign: "middle" },
  }));
  const dataRows2 = trainingInference.slice(1).map(row =>
    row.map((cell, ci) => ({
      text: cell,
      options: { fill: { color: C.bgCard }, color: C.ice, fontSize: 11, fontFace: "Calibri", align: "center", valign: "middle" },
    }))
  );

  s.addTable([headerRow2, ...dataRows2], {
    x: 1.5, y: 4.6, w: 7.0, colW: colW2,
    border: { pt: 1, color: C.bgLight },
    rowH: [0.25, 0.2, 0.2, 0.2],
  });
})();

// SLIDE 18: Normalizing Flow
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "Normalizing Flow：先验增强", "把简单高斯「揉捏」成复杂分布");

  // Problem + Solution
  const cards = [
    { title: "问题", items: ["先验 p(z|c) 是简单高斯", "不够灵活 — 语音先验分布很复杂"] },
    { title: "方案", items: ["Flow 层把高斯「揉捏」成复杂形状", "4 层交替切分通道，逐层增强"], isBest: true },
  ];

  cards.forEach((card, i) => {
    const x = 0.5 + i * 4.7;
    const fill = card.isBest ? C.bgLight : C.bgCard;
    contentCard(s, x, 1.6, 4.4, 1.2, { fill });
    if (card.isBest) {
      s.addShape(pres.shapes.RECTANGLE, { x, y: 1.6, w: 4.4, h: 0.04, fill: { color: C.amber } });
    }
    s.addText(card.title, {
      x: x + 0.25, y: 1.75, w: 3.9, h: 0.35,
      fontSize: 16, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
    });
    bulletText(s, x + 0.25, 2.15, 3.9, 0.5, card.items, { fontSize: 12 });
  });

  // Affine Coupling Layer
  s.addText("Affine Coupling Layer", {
    x: 0.5, y: 3.05, w: 9, h: 0.35,
    fontSize: 16, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  // Visual of coupling
  contentCard(s, 0.5, 3.45, 9.0, 1.8);
  s.addText([
    { text: "输入 z  →  切半  →  ", options: { color: C.ice, fontSize: 12, fontFace: "Consolas" } },
    { text: "z₁", options: { color: C.amber, fontSize: 12, fontFace: "Consolas", bold: true } },
    { text: "  直接通过\n", options: { color: C.ice, fontSize: 12, fontFace: "Consolas", breakLine: true } },
    { text: "               →  ", options: { color: C.ice, fontSize: 12, fontFace: "Consolas" } },
    { text: "z₁", options: { color: C.amber, fontSize: 12, fontFace: "Consolas", bold: true } },
    { text: "  →  NN  →  scale, shift\n", options: { color: C.ice, fontSize: 12, fontFace: "Consolas", breakLine: true } },
    { text: "               →  ", options: { color: C.ice, fontSize: 12, fontFace: "Consolas" } },
    { text: "z₂' = z₂ × exp(scale) + shift\n", options: { color: C.amber, fontSize: 12, fontFace: "Consolas", breakLine: true } },
    { text: "拼接  z₁ + z₂'  →  输出", options: { color: C.ice, fontSize: 12, fontFace: "Consolas" } },
  ], { x: 0.8, y: 3.5, w: 8.4, h: 1.3, valign: "middle", margin: 8 });

  s.addText("关键优势：可逆 — 反向只需算术逆，NN 不需要可逆", {
    x: 0.5, y: 4.85, w: 9, h: 0.25,
    fontSize: 13, fontFace: "Calibri", color: C.amber, italic: true, align: "center", margin: 0,
  });
})();

// ================================================================
// SLIDE 14: KL Loss
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "KL Loss：训练的核心推动力", "让先验学会后验 — 知识蒸馏");

  const cols = [
    {
      title: "后验路径（训练「标准答案」）",
      color: C.green,
      items: ["真实音频 → PosteriorEncoder → q(z|x)", "质量好，但推理时不可用"],
    },
    {
      title: "先验路径（推理时可用）",
      color: C.amber,
      items: ["文本 → TextEncoder → p(z|c)", "推理时使用，但刚开始不准"],
    },
  ];

  cols.forEach((col, i) => {
    const x = 0.5 + i * 4.7;
    contentCard(s, x, 1.6, 4.4, 1.5);
    accentBar(s, x, 1.6, 0.06, 1.5);
    s.addText(col.title, {
      x: x + 0.25, y: 1.7, w: 3.9, h: 0.35,
      fontSize: 15, fontFace: "Georgia", color: col.color, bold: true, margin: 0,
    });
    bulletText(s, x + 0.25, 2.1, 3.9, 0.8, col.items, { fontSize: 12 });
  });

  // KL formula
  s.addText("KL(q(z|x) || p(z|c)) → 越小 → 先验越接近后验 → 推理质量越高", {
    x: 0.5, y: 3.3, w: 9, h: 0.4,
    fontSize: 15, fontFace: "Georgia", color: C.white, align: "center", bold: true, margin: 0,
  });

  // Comparison
  contentCard(s, 0.5, 3.9, 9.0, 1.4);
  accentBar(s, 0.5, 3.9, 0.06, 1.4);
  s.addText("对比 Tacotron2", {
    x: 0.8, y: 4.0, w: 8.4, h: 0.35,
    fontSize: 15, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });
  bulletText(s, 0.8, 4.4, 8.4, 0.55, [
    "Tacotron2 训练时没有「后验路径」— 直接在预测 Mel 上算 L1 loss",
    "VITS 的 KL Loss 让先验和后验在隐空间层面做匹配，比纯输出层监督更本质",
  ], { fontSize: 12 });
})();

// ================================================================
// SLIDE 19: VITS 后续影响
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "VITS 的后续影响", "TTS 分岔为两条路线");

  // Tree
  s.addText("VITS  (VAE + Flow + 端到端)", {
    x: 2.8, y: 1.6, w: 4.4, h: 0.6,
    fontSize: 18, fontFace: "Georgia", color: C.white, bold: true, align: "center", margin: 0,
  });
  flowBox(s, 2.8, 1.55, 4.4, 0.65, "", { fill: C.bgLight });

  // Split lines
  s.addText("┌", { x: 3.2, y: 2.15, w: 1.2, h: 0.5, fontSize: 20, fontFace: "Consolas", color: C.amber, align: "center", margin: 0 });
  s.addText("┐", { x: 5.6, y: 2.15, w: 1.2, h: 0.5, fontSize: 20, fontFace: "Consolas", color: C.amber, align: "center", margin: 0 });

  const branches = [
    { title: "CosyVoice", sub: "Flow Matching + 连续表示", desc: "质量上限高，架构自由\n可用 Transformer", x: 0.5 },
    { title: "FishSpeech", sub: "离散 token + LLM", desc: "享受 Scaling Law\n可扩展", x: 5.2 },
  ];

  branches.forEach(br => {
    contentCard(s, br.x, 2.7, 4.4, 1.6);
    s.addShape(pres.shapes.RECTANGLE, {
      x: br.x, y: 2.7, w: 4.4, h: 0.04, fill: { color: C.amber },
    });
    s.addText(br.title, {
      x: br.x + 0.2, y: 2.85, w: 4.0, h: 0.4,
      fontSize: 20, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
    });
    s.addText(br.sub, {
      x: br.x + 0.2, y: 3.25, w: 4.0, h: 0.3,
      fontSize: 14, fontFace: "Calibri", color: C.amber, margin: 0,
    });
    s.addText(br.desc, {
      x: br.x + 0.2, y: 3.6, w: 4.0, h: 0.55,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });

  // Three evolution lines
  s.addText("三条路线演化", {
    x: 0.5, y: 4.3, w: 9, h: 0.3,
    fontSize: 14, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  const evolutions = [
    "Tacotron2 → VITS：AR→NAR，两阶段→端到端，MAS 解决对齐",
    "VITS → CosyVoice：Flow→Flow Matching，可用 Transformer",
    "VITS → FishSpeech：连续隐变量→离散 token，接入 LLM 生态",
  ];

  bulletText(s, 0.5, 4.65, 9.0, 0.4, evolutions, { fontSize: 11 });
})();

// ================================================================
// SLIDE 20: 三阶段技术接力
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  slideTitle(s, "三阶段技术接力", "总结");

  const headers = ["阶段", "输入 → 输出", "核心问题", "解决方案"];
  const rows = [
    ["Mel 频谱", "波形 → Mel", "波形太「死板」，维度高", "Mel 感知压缩，80维"],
    ["HiFi-GAN", "Mel → 波形", "GL 重建质量差", "GAN + MPD/MSD 双判别器"],
    ["VITS", "文本 → 波形", "AR 误差 + 对齐不稳定", "MAS 对齐 + NAR + VAE"],
  ];

  const colW = [1.3, 1.6, 2.4, 3.7];
  const headerRow = headers.map(h => ({
    text: h,
    options: { fill: { color: C.bgLight }, color: C.white, bold: true, fontSize: 14, fontFace: "Calibri", align: "center", valign: "middle" },
  }));
  const dataRows = rows.map((row, ri) =>
    row.map((cell, ci) => ({
      text: cell,
      options: {
        fill: { color: ri === 2 ? C.bgLight : C.bgCard },
        color: ri === 2 ? C.amber : C.ice,
        bold: ri === 2 || ci === 3,
        fontSize: 13, fontFace: "Calibri", align: "center", valign: "middle",
      },
    }))
  );

  s.addTable([headerRow, ...dataRows], {
    x: 0.5, y: 1.6, w: 9.0, colW,
    border: { pt: 1, color: C.bgLight },
    rowH: [0.5, 0.55, 0.55, 0.55],
  });

  // Complete pipeline
  s.addText("完整链路", {
    x: 0.5, y: 3.65, w: 9, h: 0.35,
    fontSize: 16, fontFace: "Georgia", color: C.white, bold: true, margin: 0, align: "center",
  });

  s.addText("「大家好」 → 音素 /d/a/j/i/a/h/a/o/ → TextEncoder → z → HiFi-GAN → 波形 ✓", {
    x: 0.5, y: 4.1, w: 9, h: 0.45,
    fontSize: 14, fontFace: "Consolas", color: C.amber, align: "center", margin: 0,
  });

  s.addText("VITS 用隐变量 z 替代 Mel — TTS 从「手工设计」走向「端到端学习」的分水岭", {
    x: 0.5, y: 4.5, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.white, align: "center", italic: true, margin: 0,
  });
})();

// ================================================================
// SLIDE 21: Thank You
// ================================================================
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Decorative waveform at top
  for (let i = 0; i < 12; i++) {
    const h = 0.08 + Math.abs(Math.sin(i * 0.7)) * 0.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5 + i * 0.78, y: 0.3, w: 0.06, h,
      fill: { color: C.amber, transparency: 70 - i * 3 },
    });
  }

  // Left amber accent
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 1.8, w: 0.08, h: 2.0,
    fill: { color: C.amber },
  });

  s.addText("Thank You", {
    x: 0.6, y: 1.8, w: 8.8, h: 0.8,
    fontSize: 48, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  s.addText("从 Mel 频谱到 VITS——TTS 学习之路", {
    x: 0.6, y: 2.7, w: 8.8, h: 0.5,
    fontSize: 20, fontFace: "Georgia", color: C.amber, margin: 0,
  });

  s.addText("下次分享：LLM-based TTS 原理", {
    x: 0.6, y: 3.5, w: 8.8, h: 0.5,
    fontSize: 16, fontFace: "Calibri", color: C.muted, italic: true, margin: 0,
  });

  s.addText("Q & A", {
    x: 0.6, y: 4.2, w: 8.8, h: 0.5,
    fontSize: 24, fontFace: "Georgia", color: C.white, margin: 0,
  });
})();

// ── Generate ──
(async () => {
  const outPath = "/mist/dengliang/LearnTTS/stage3_vits/output/TTS_Talk_SignalFlow.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log("PPT saved to: " + outPath);
})();
