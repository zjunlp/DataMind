import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { useGSAP } from "@gsap/react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  ArrowDown,
  ArrowUpRight,
  Check,
  ChevronDown,
  Copy,
  FileText,
  Github,
  Languages,
  Mail,
  Menu,
  X,
} from "lucide-react";
import "@fontsource-variable/manrope";
import "@fontsource-variable/newsreader";
import "@fontsource/dm-mono/400.css";
import domainDistributionFigure from "./assets/longds-domain-distribution.png";
import performanceDegradationFigure from "./assets/longds-performance-degradation.png";
import "./styles.css";

gsap.registerPlugin(useGSAP, ScrollTrigger);

const PAPER_URL = "https://arxiv.org/abs/2605.30434";
const REPO_URL = "https://github.com/zjunlp/DataMind";
const LONGDS_DOC_URL = "https://github.com/zjunlp/DataMind/tree/main/longds";
const CONTACT_EMAIL = "zhangningyu@zju.edu.cn";
const VISITOR_API_URL = "https://bsz.iirose.cn/api";
const VISITOR_ID_KEY = "longds_busuanzi_identity";
const StateAtlas = React.lazy(() => import("./StateAtlas"));

const HERO_STATS = [
  { value: 68, decimals: 0, suffix: "" },
  { value: 2225, decimals: 0, suffix: "" },
  { value: 11.29, decimals: 2, suffix: "" },
];

const QUICK_START_COMMANDS = [
  {
    key: "environment",
    commands: [
      "cd DataMind/longds/runners/DSGym",
      "uv sync",
    ],
  },
  {
    key: "dataset",
    commands: [
      "cd /path/to/DataMind/longds",
      `hf download zjunlp/LongDS \\
  --repo-type dataset \\
  --local-dir dataset`,
    ],
  },
  {
    key: "executors",
    commands: [
      "cd DataMind/longds/runners/DSGym/executors",
      "docker build -t executor-prebuilt ./container_images/longds_image",
      "docker build -t manager-prebuilt ./manager",
      `python generate_compose.py \\
  -n 16 \\
  --types "executor-prebuilt:16" \\
  -m ../../../dataset/data`,
      "docker compose -f docker-compose.yml up -d --build",
    ],
  },
  {
    key: "evaluation",
    commands: [
      "cd DataMind/longds/runners/DSGym/scripts",
      `uv run python longds.py \\
  --dataset longds \\
  --model openai/gpt-5.4 \\
  --backend litellm \\
  --output-dir ./results`,
    ],
  },
];

let visitorCountRequest;

function formatStatValue(value, decimals, suffix) {
  return `${value.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}${suffix}`;
}

function HeroStats({ label, labels }) {
  const statsRef = useRef(null);

  useGSAP(() => {
    const counters = gsap.utils.toArray("[data-stat-value]", statsRef.current);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    counters.forEach((element) => {
      const value = Number(element.dataset.statValue);
      const decimals = Number(element.dataset.statDecimals);
      element.textContent = formatStatValue(reduceMotion ? value : 0, decimals, element.dataset.statSuffix);
    });

    if (reduceMotion) return;

    const timeline = gsap.timeline({
      scrollTrigger: {
        trigger: statsRef.current,
        start: "top 90%",
        once: true,
      },
    });

    counters.forEach((element, index) => {
      const counter = { value: 0 };
      const value = Number(element.dataset.statValue);
      const decimals = Number(element.dataset.statDecimals);
      const suffix = element.dataset.statSuffix;

      timeline.to(counter, {
        value,
        duration: 1.35,
        ease: "power3.out",
        onUpdate: () => {
          element.textContent = formatStatValue(counter.value, decimals, suffix);
        },
      }, index * 0.1);
    });
  }, { scope: statsRef });

  return (
    <dl className="hero-stats" aria-label={label} ref={statsRef}>
      {HERO_STATS.map(({ value, decimals, suffix }, index) => (
        <div key={value}>
          <dt
            data-stat-value={value}
            data-stat-decimals={decimals}
            data-stat-suffix={suffix}
          >
            {formatStatValue(value, decimals, suffix)}
          </dt>
          <dd>{labels[index]}</dd>
        </div>
      ))}
    </dl>
  );
}

async function writeToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through for browsers that restrict the asynchronous clipboard API.
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  return copied;
}

function QuickStart({ content }) {
  const [activeKey, setActiveKey] = useState(QUICK_START_COMMANDS[0].key);
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef(null);
  const activeCommand = QUICK_START_COMMANDS.find(({ key }) => key === activeKey) ?? QUICK_START_COMMANDS[0];

  useEffect(() => () => window.clearTimeout(copyTimerRef.current), []);

  const selectCommand = (key) => {
    window.clearTimeout(copyTimerRef.current);
    setCopied(false);
    setActiveKey(key);
  };

  const copyCommand = async () => {
    if (!await writeToClipboard(activeCommand.commands.join("\n\n"))) return;
    setCopied(true);
    window.clearTimeout(copyTimerRef.current);
    copyTimerRef.current = window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <section className="quickstart-section" id="quick-start" aria-labelledby="quickstart-title">
      <h2 id="quickstart-title">{content.title}</h2>
      <div className="quickstart-grid">
        <div className="quick-terminal" aria-label={content.terminalLabel}>
          <div className="quick-terminal-head">
            <div className="quick-terminal-title">
              <span className="quick-terminal-dots" aria-hidden="true"><span /><span /><span /></span>
              <span>LongDS-Bench</span>
            </div>
            <button
              className={copied ? "quick-copy is-copied" : "quick-copy"}
              type="button"
              onClick={copyCommand}
              aria-label={copied ? content.copied : content.copy}
              title={copied ? content.copied : content.copy}
            >
              {copied ? <Check size={19} /> : <Copy size={19} />}
            </button>
          </div>
          <div className="quick-terminal-content">
            <div className="quick-terminal-tabs" role="tablist" aria-label={content.commandSets}>
              {QUICK_START_COMMANDS.map(({ key }, index) => (
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeKey === key}
                  aria-controls="quick-terminal-output"
                  id={`quick-terminal-tab-${key}`}
                  className={activeKey === key ? "is-active" : ""}
                  key={key}
                  onClick={() => selectCommand(key)}
                >
                  <span aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
                  {content.tabs[index]}
                </button>
              ))}
            </div>
            <div
              className="quick-terminal-output"
              id="quick-terminal-output"
              role="tabpanel"
              aria-labelledby={`quick-terminal-tab-${activeKey}`}
            >
              {activeCommand.commands.map((command) => (
                <pre className="quick-command" key={command}><code>{command}</code></pre>
              ))}
            </div>
          </div>
        </div>

        <aside className="quickstart-guide">
          <div className="quickstart-guide-head">
            <span className="quickstart-guide-icon"><Github size={20} aria-hidden="true" /></span>
            <div>
              <h3>{content.prerequisites}</h3>
              <ul className="quickstart-requirements">
                <li>Python 3.12</li>
                <li>Docker and Docker Compose</li>
                <li><code>uv</code></li>
              </ul>
            </div>
          </div>
          <p>{content.description}</p>
          <a className="quickstart-docs" href={LONGDS_DOC_URL} target="_blank" rel="noreferrer">
            <Github size={18} aria-hidden="true" />
            {content.documentation}
          </a>
        </aside>
      </div>
    </section>
  );
}

function requestVisitorCount() {
  if (!visitorCountRequest) {
    const identity = window.localStorage.getItem(VISITOR_ID_KEY);
    const headers = { "x-bsz-referer": window.location.href };
    if (identity) headers.Authorization = `Bearer ${identity}`;

    visitorCountRequest = fetch(VISITOR_API_URL, { method: "POST", headers })
      .then(async (response) => {
        if (!response.ok) return null;

        const nextIdentity = response.headers.get("Set-Bsz-Identity");
        if (nextIdentity) window.localStorage.setItem(VISITOR_ID_KEY, nextIdentity);

        const result = await response.json();
        return result.success ? result.data.site_pv : null;
      })
      .catch(() => null);
  }

  return visitorCountRequest;
}

const models = [
  {
    model: "Gemini-3.1-Pro",
    harness: "DSGym",
    org: "Google",
    cost: "—",
    date: "2026-09-05",
    type: "proprietary",
    avgSteps: 117.82,
    scores: {
      overall: 48.45,
      education: 58.03,
      community: 69.54,
      socialGood: 41.73,
      business: 33.59,
      geoscience: 42.2,
      sports: 31.85,
    },
  },
  {
    model: "GPT-5.4",
    harness: "DSGym",
    org: "OpenAI",
    cost: "—",
    date: "2026-09-05",
    type: "proprietary",
    avgSteps: 68.57,
    scores: {
      overall: 43.5,
      education: 77.92,
      community: 65.32,
      socialGood: 36.8,
      business: 28.4,
      geoscience: 28.9,
      sports: 10.52,
    },
  },
  {
    model: "Claude-4.6-Sonnet",
    harness: "DSGym",
    org: "Anthropic",
    cost: "—",
    date: "2026-09-05",
    type: "proprietary",
    avgSteps: 170.04,
    scores: {
      overall: 41.56,
      education: 77.29,
      community: 54.64,
      socialGood: 36.1,
      business: 25.54,
      geoscience: 31.92,
      sports: 19.76,
    },
  },
  {
    model: "Kimi-K2.6",
    harness: "DSGym",
    org: "Moonshot AI",
    cost: "—",
    date: "2026-09-05",
    type: "open",
    avgSteps: 115.41,
    scores: {
      overall: 39.72,
      education: 64.98,
      community: 60.62,
      socialGood: 31.29,
      business: 20.99,
      geoscience: 28.83,
      sports: 32.85,
    },
  },
  {
    model: "DeepSeek-V4-Pro",
    harness: "DSGym",
    org: "DeepSeek AI",
    cost: "—",
    date: "2026-09-05",
    type: "open",
    avgSteps: 133.12,
    scores: {
      overall: 31.97,
      education: 61.36,
      community: 49.47,
      socialGood: 32.41,
      business: 17.06,
      geoscience: 16.6,
      sports: 15.82,
    },
  },
];

const v11FullModels = [
  {
    model: "Claude Fable 5.1",
    harness: "Claude Code",
    org: "Anthropic",
    date: "2026-09-05",
    type: "proprietary",
    ranked: false,
    unrankedOrder: 2,
    note: "(24-task Lite only)",
    cost: "¥3,868.32",
    costPerTask: "¥161.18 / task",
    scores: { overall: 76.53, education: 88.21, community: 72.53, socialGood: 86.82, business: 85.91, geoscience: 82.81, sports: 38.25 },
  },
  {
    model: "DeepSeek V4 Pro",
    harness: "Claude Code",
    org: "DeepSeek AI",
    date: "2026-09-05",
    type: "open",
    cost: "¥731.96",
    costPerTask: "¥10.76 / task",
    scores: { overall: 40.96, education: 72.95, community: 57.54, socialGood: 36.22, business: 16.59, geoscience: 34.3, sports: 22.74 },
  },
  {
    model: "GLM-5.2",
    harness: "Claude Code",
    org: "Z.ai",
    date: "2026-09-05",
    type: "open",
    cost: "¥6,943.46",
    costPerTask: "¥102.11 / task",
    scores: { overall: 54.25, education: 88.91, community: 74.47, socialGood: 48.58, business: 33.61, geoscience: 42.51, sports: 29.74 },
  },
  {
    model: "GPT-5.6-sol",
    harness: "Codex",
    org: "OpenAI",
    date: "2026-09-05",
    type: "proprietary",
    cost: "¥4,004.11",
    costPerTask: "¥58.88 / task",
    scores: { overall: 64.42, education: 93.87, community: 77.06, socialGood: 55.36, business: 56.99, geoscience: 56.18, sports: 30.58 },
  },
  {
    model: "GPT-6 Astra",
    harness: "Codex",
    org: "OpenAI",
    date: "2026-09-05",
    type: "proprietary",
    ranked: false,
    note: "(24-task Lite only)",
    cost: "¥3,040.84",
    costPerTask: "¥126.70 / task",
    scores: { overall: 78.17, education: 96.33, community: 63.09, socialGood: 81.31, business: 88.72, geoscience: 81.32, sports: 67.73 },
  },
  {
    model: "Kimi K3",
    harness: "Kimi Code",
    org: "Moonshot AI",
    date: "2026-09-05",
    type: "open",
    cost: "¥1,442.35",
    costPerTask: "¥21.21 / task",
    scores: { overall: 58.59, education: 82.2, community: 76, socialGood: 56.12, business: 43.35, geoscience: 46.02, sports: 51.64 },
  },
  {
    model: "Qwen3.8-Max-0803",
    harness: "Qoder",
    org: "Qwen",
    date: "2026-09-05",
    type: "proprietary",
    cost: "7,722.302",
    costPerTask: "113.563 / task",
    costUnit: "credits",
    scores: { overall: 49.86, education: 77.06, community: 72.28, socialGood: 43.66, business: 29.4, geoscience: 39.39, sports: 26.48 },
  },
  {
    model: "Qwen3.8-Max-0902",
    harness: "Qoder",
    org: "Qwen",
    date: "2026-09-05",
    type: "proprietary",
    cost: "26,426.995",
    costPerTask: "388.632 / task",
    costUnit: "credits",
    scores: { overall: 56.88, education: 88.88, community: 76.87, socialGood: 51.21, business: 41.96, geoscience: 43.85, sports: 26.09 },
  },
];

const v11LiteModels = [
  {
    model: "Claude Fable 5.1",
    harness: "Claude Code",
    org: "Anthropic",
    date: "2026-09-05",
    type: "proprietary",
    cost: "¥3,868.32",
    costPerTask: "¥161.18 / task",
    scores: { overall: 76.53, education: 88.21, community: 72.53, socialGood: 86.82, business: 85.91, geoscience: 82.81, sports: 38.25 },
  },
  {
    model: "DeepSeek V4 Pro",
    harness: "Claude Code",
    org: "DeepSeek AI",
    date: "2026-09-05",
    type: "open",
    cost: "¥226.52",
    costPerTask: "¥9.44 / task",
    scores: { overall: 39.33, education: 54.83, community: 33.52, socialGood: 53.2, business: 18.94, geoscience: 55.4, sports: 22.74 },
  },
  {
    model: "GLM-5.2",
    harness: "Claude Code",
    org: "Z.ai",
    date: "2026-09-05",
    type: "open",
    cost: "¥2,260.56",
    costPerTask: "¥94.19 / task",
    scores: { overall: 56.59, education: 78.21, community: 62.96, socialGood: 79.86, business: 42.25, geoscience: 42.01, sports: 29.74 },
  },
  {
    model: "GPT-5.6-sol",
    harness: "Codex",
    org: "OpenAI",
    date: "2026-09-05",
    type: "proprietary",
    cost: "¥1,194.39",
    costPerTask: "¥49.77 / task",
    scores: { overall: 70.71, education: 90.35, community: 62.9, socialGood: 76.78, business: 84.54, geoscience: 77.86, sports: 30.58 },
  },
  {
    model: "GPT-6 Astra",
    harness: "Codex",
    org: "OpenAI",
    date: "2026-09-05",
    type: "proprietary",
    cost: "¥3,040.84",
    costPerTask: "¥126.70 / task",
    scores: { overall: 78.17, education: 96.33, community: 63.09, socialGood: 81.31, business: 88.72, geoscience: 81.32, sports: 67.73 },
  },
  {
    model: "Kimi K3",
    harness: "Kimi Code",
    org: "Moonshot AI",
    date: "2026-09-05",
    type: "open",
    cost: "¥455.40",
    costPerTask: "¥18.98 / task",
    scores: { overall: 68.9, education: 71.88, community: 66.6, socialGood: 84.75, business: 56.15, geoscience: 79.95, sports: 51.64 },
  },
  {
    model: "Qwen3.8-Max-0803",
    harness: "Qoder",
    org: "Qwen",
    date: "2026-09-05",
    type: "proprietary",
    cost: "2,414.182",
    costPerTask: "100.591 / task",
    costUnit: "credits",
    scores: { overall: 54.6, education: 57.97, community: 67.02, socialGood: 63.38, business: 41.8, geoscience: 58.55, sports: 26.48 },
  },
  {
    model: "Qwen3.8-Max-0902",
    harness: "Qoder",
    org: "Qwen",
    date: "2026-09-05",
    type: "proprietary",
    cost: "8,604.453",
    costPerTask: "358.519 / task",
    costUnit: "credits",
    scores: { overall: 62.58, education: 83.77, community: 71.98, socialGood: 73.29, business: 51.94, geoscience: 59.9, sports: 26.09 },
  },
];

// Keep benchmark releases separate so a new release can ship its own scores
// without changing the published snapshot for the original benchmark.
const leaderboardModels = {
  v1_1_lite: v11LiteModels,
  v1_1_full: v11FullModels,
  v1: models,
};

const leaderboardDomainCounts = {
  v1_1_lite: { education: 3, community: 6, socialGood: 4, business: 4, geoscience: 4, sports: 3 },
  v1_1_full: { education: 8, community: 16, socialGood: 10, business: 12, geoscience: 19, sports: 3 },
};

const translations = {
  en: {
    pageTitle: "LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis",
    skip: "Skip to leaderboard",
    home: "LongDS home",
    openNav: "Open navigation",
    closeNav: "Close navigation",
    mainNav: "Main navigation",
    switchLanguage: "Switch to Chinese",
    languageLabel: "Chinese",
    nav: { leaderboard: "Leaderboard", benchmark: "LongDS Benchmark", findings: "Deep Analysis", quickStart: "Quick Start" },
    repository: "Repository",
    hero: {
      thesis: "LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis",
      summary: "We introduce LongDS, a benchmark for long-horizon, multi-turn data analysis where agents must maintain, update, restore, and compose evolving analytical states.",
      explore: "Leaderboard",
      paper: "Paper",
      statsLabel: "Benchmark statistics",
      stats: ["Tasks", "Turns", "Dependency span / turn"],
      stateScene: {
        heading: "Evolving Analytical States",
        turn1: "Calculate X on the cleaned data ...",
        turn2: "Filter data ...",
        turn3: "Update the definition...",
        turn4: "Use initial data...",
        turn5: "Calculate on new data using the best combination ...",
        nodes: {
          clean: "clean(raw_data)",
          filter: "filter(df)",
          mean: "calc_X = mean",
          median: "calc_X = median",
          agent: "select correct state",
          initial: "initial data",
          newData: "new_data",
          composition: "best combination",
        },
      },
      stateFigureLabel: "Multi-turn, long-horizon analytical state management in LongDS. Agents track evolving filters, definitions, and intermediate results to select the correct state for requests depending on prior turns.",
      figureCaption: "Domain and task distribution of LongDS. The inner ring shows application domains, while the outer ring shows source datasets and Kaggle competitions, with sector size proportional to the number of long-horizon analysis tasks.",
    },
    domains: {
      overall: "Overall", education: "Education", community: "Community", socialGood: "Social Good",
      business: "Business", geoscience: "Geoscience", sports: "Sports",
    },
    leaderboard: {
      title: "Leaderboard",
      contact: "Get listed",
      filterLabel: "Model type filter",
      filters: { all: "All Models", proprietary: "Proprietary Models", open: "Open-source Models" },
      versionLabel: "Benchmark version",
      versions: { v1_1_lite: "LongDS v1.1 Lite", v1_1_full: "LongDS v1.1 Full", v1: "LongDS v1" },
      emptyState: "No models match the current filters.",
      domainLabel: "Score domain",
      columns: { rank: "Rank", model: "Model", harness: "Harness", score: "Score", cost: "Cost", org: "Org", date: "Date" },
      types: { open: "Open-source", proprietary: "Proprietary" },
    },
    benchmark: {
      title: "State-evolution patterns in LongDS",
      patterns: [
        ["Initial", "Establishes a reusable analytical object, such as a cohort, metric, rule, or intermediate result."],
        ["Inheritance", "Reuses the most recent valid analytical state without restating it."],
        ["Update", "Revises a previous definition, formula, filter, aggregation rule, or baseline, making the revision the new default state."],
        ["Counterfactual", "Introduces a temporary alternative assumption for the current turn only."],
        ["Rollback", "Answers under an earlier anchored version of the analysis instead of the most recent state."],
        ["Composition", "Combines two or more explicit state operations beyond default inheritance."],
      ],
    },
    findings: {
      title: "Long-horizon performance degradation in LongDS",
      figureCaption: "Long-horizon performance degradation in LongDS. Accuracy drops across three increasing demands: (a) later task progress, averaged within each 10% progress interval; (b) larger dependency breadth, with n denoting the number of turns per group; and (c) more complex state-evolution patterns.",
      metrics: [
        ["−46.8 pts", "Long-Horizon Performance", "Accuracy decreases as tasks progress."],
        ["52%–69%", "Long-Horizon Errors", "Long-horizon errors account for the majority of failures, ranging from 52% for GPT-5.4 to 69% for Kimi-K2.6."],
        ["2.85", "Dependency Breadth / Turn", "Task-level mean number of direct prior-turn dependencies per turn."],
      ],
    },
    quickStart: {
      title: "Quick Start",
      terminalLabel: "Quick start terminal",
      commandSets: "Quick start command sets",
      tabs: ["Environment Setup", "Data", "Execution Environment", "Run Evaluation"],
      prerequisites: "Prerequisites",
      description: "The paper experiments use DSGym, which provides Docker-based execution infrastructure for code-based data analysis.",
      documentation: "GitHub",
      copy: "Copy current commands",
      copied: "Copied",
    },
    views: "Views",
  },
  zh: {
    pageTitle: "LongDS-Bench：关于长程智能体数据分析的失败",
    skip: "跳到排行榜",
    home: "LongDS 首页",
    openNav: "打开导航",
    closeNav: "关闭导航",
    mainNav: "主导航",
    switchLanguage: "切换到英文",
    languageLabel: "EN",
    nav: { leaderboard: "排行榜", benchmark: "LongDS 评测基准", findings: "深度分析", quickStart: "快速开始" },
    repository: "代码仓库",
    hero: {
      thesis: "LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis",
      summary: "我们提出 LongDS，这是一个面向长程、多轮数据分析的评测基准，要求智能体维护、更新、恢复并组合不断演化的分析状态。",
      explore: "排行榜",
      paper: "论文",
      statsLabel: "评测统计",
      stats: ["任务数", "轮次数", "每轮依赖跨度"],
      stateScene: {
        heading: "演化中的分析状态",
        turn1: "在清洗后的数据上计算 X……",
        turn2: "筛选数据……",
        turn3: "更新定义……",
        turn4: "使用初始数据……",
        turn5: "使用最佳组合在新数据上计算……",
        nodes: {
          clean: "clean(raw_data)",
          filter: "filter(df)",
          mean: "calc_X = mean",
          median: "calc_X = median",
          agent: "选择正确状态",
          initial: "初始数据",
          newData: "new_data",
          composition: "最佳组合",
        },
      },
      stateFigureLabel: "LongDS 中的多轮、长程分析状态管理。智能体跟踪不断演化的筛选条件、定义和中间结果，为依赖先前轮次的请求选择正确的状态。",
      figureCaption: "LongDS 的领域与任务分布。内环表示应用领域，外环表示源数据集和 Kaggle 竞赛，扇区大小与长程分析任务数量成正比。",
    },
    domains: {
      overall: "综合", education: "教育", community: "社区", socialGood: "社会公益",
      business: "商业", geoscience: "地球科学", sports: "体育",
    },
    leaderboard: {
      title: "排行榜",
      contact: "联系上榜",
      filterLabel: "模型类型筛选",
      filters: { all: "全部模型", proprietary: "专有模型", open: "开源模型" },
      versionLabel: "评测版本",
      versions: { v1_1_lite: "LongDS v1.1 Lite", v1_1_full: "LongDS v1.1 Full", v1: "LongDS v1" },
      emptyState: "没有符合当前筛选条件的模型。",
      domainLabel: "得分领域",
      columns: { rank: "排名", model: "模型", harness: "运行框架", score: "得分", cost: "成本", org: "机构", date: "日期" },
      types: { open: "开源", proprietary: "专有" },
    },
    benchmark: {
      title: "LongDS 中的状态演化模式",
      patterns: [
        ["初始", "建立一个可复用的分析对象，例如用户群体、指标、规则或中间结果。"],
        ["继承", "无需重述，复用最近一次有效的分析状态。"],
        ["更新", "修改先前的定义、公式、筛选条件、聚合规则或基线，并将修改后的版本设为新的默认状态。"],
        ["反事实", "仅在当前轮次中引入一个临时的替代假设。"],
        ["回滚", "基于先前锚定的分析版本，而不是最新状态回答问题。"],
        ["组合", "在默认继承之外，组合两个或更多明确的状态操作。"],
      ],
    },
    findings: {
      title: "LongDS 中的长程性能退化",
      figureCaption: "LongDS 中的长程性能退化。准确率随着三类需求增加而下降：（a）任务进度后移，以每 10% 的进度区间取平均；（b）依赖广度增大，其中 n 表示每组的轮次数；（c）状态演化模式更加复杂。",
      metrics: [
        ["−46.8 分", "长程性能", "准确率随着任务推进而下降。"],
        ["52%–69%", "长程错误", "长程错误占失败案例的大多数：GPT-5.4 为 52%，Kimi-K2.6 为 69%。"],
        ["2.85", "每轮依赖广度", "每轮直接依赖的先前轮次数量在任务级别上的平均值。"],
      ],
    },
    quickStart: {
      title: "快速开始",
      terminalLabel: "快速开始终端",
      commandSets: "快速开始命令组",
      tabs: ["环境", "数据集", "执行器", "评测"],
      prerequisites: "前置条件",
      description: "论文实验使用 DSGym，它为基于代码的数据分析提供基于 Docker 的执行基础设施。",
      documentation: "GitHub",
      copy: "复制当前命令",
      copied: "已复制",
    },
    views: "访问量",
  },
};

function App() {
  const [language, setLanguage] = useState("en");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [leaderboardVersion, setLeaderboardVersion] = useState("v1_1_lite");
  const [modelType, setModelType] = useState("all");
  const [domain, setDomain] = useState("overall");
  const [visitorCount, setVisitorCount] = useState(null);
  const t = translations[language];

  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.title = t.pageTitle;
  }, [language, t.pageTitle]);

  useEffect(() => {
    let active = true;
    requestVisitorCount().then((count) => {
      if (active && Number.isFinite(count)) setVisitorCount(count);
    });

    return () => {
      active = false;
    };
  }, []);

  const filteredModels = useMemo(() => {
    return leaderboardModels[leaderboardVersion]
      .filter((item) => modelType === "all" || item.type === modelType)
      .sort((a, b) => {
        if (a.ranked === false || b.ranked === false) {
          if (a.ranked === b.ranked) {
            return (a.unrankedOrder ?? 0) - (b.unrankedOrder ?? 0);
          }
          return a.ranked === false ? 1 : -1;
        }
        return b.scores[domain] - a.scores[domain];
      });
  }, [domain, modelType, leaderboardVersion]);

  const maxScore = filteredModels.length > 0
    ? Math.max(...filteredModels.map((item) => item.scores[domain]))
    : 100;
  const domainCounts = leaderboardDomainCounts[leaderboardVersion];
  const domainDisplayName = (value, label) => {
    const count = domainCounts?.[value];
    return count ? `${label} (${count})` : label;
  };

  const toggleLanguage = () => {
    setLanguage((current) => (current === "en" ? "zh" : "en"));
  };

  const closeMobile = () => setMobileOpen(false);

  return (
    <div className="app" data-language={language}>
      <a className="skip-link" href="#leaderboard">
        {t.skip}
      </a>

      <header className="site-header">
        <a className="wordmark" href="#top" aria-label={t.home} onClick={closeMobile}>
          <span className="wordmark-mark" aria-hidden="true">
            L<span>↺</span>
          </span>
          <span>LongDS</span>
        </a>

        <button
          className="icon-button menu-button"
          type="button"
          onClick={() => setMobileOpen((open) => !open)}
          aria-label={mobileOpen ? t.closeNav : t.openNav}
          aria-expanded={mobileOpen}
        >
          {mobileOpen ? <X size={19} /> : <Menu size={19} />}
        </button>

        <nav className={mobileOpen ? "site-nav is-open" : "site-nav"} aria-label={t.mainNav}>
          <a href="#leaderboard" onClick={closeMobile}>{t.nav.leaderboard}</a>
          <a href="#benchmark" onClick={closeMobile}>{t.nav.benchmark}</a>
          <a href="#findings" onClick={closeMobile}>{t.nav.findings}</a>
          <a href="#quick-start" onClick={closeMobile}>{t.nav.quickStart}</a>
          <button className="mobile-language-toggle" type="button" aria-label={t.switchLanguage} onClick={() => { toggleLanguage(); closeMobile(); }}>
            <Languages size={18} /> {t.languageLabel}
          </button>
        </nav>

        <div className="header-actions">
          <button className="button button-quiet button-compact language-toggle" type="button" onClick={toggleLanguage} aria-label={t.switchLanguage}>
            {t.languageLabel}
          </button>
          <a className="button button-dark button-compact" href={REPO_URL} target="_blank" rel="noreferrer">
            <Github size={17} /> {t.repository}
          </a>
        </div>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-copy">
            <h1 id="hero-title">LongDS-Bench</h1>
            <p className="hero-thesis">{t.hero.thesis}</p>
            <p className="hero-summary">{t.hero.summary}</p>
            <div className="hero-actions">
              <a className="button button-primary" href="#leaderboard">
                {t.hero.explore} <ArrowDown size={16} />
              </a>
              <a className="button button-quiet" href={PAPER_URL} target="_blank" rel="noreferrer">
                <FileText size={16} /> {t.hero.paper}
              </a>
            </div>
            <HeroStats label={t.hero.statsLabel} labels={t.hero.stats} />
          </div>
          <React.Suspense fallback={null}>
            <StateAtlas content={t.hero.stateScene} label={t.hero.stateFigureLabel} />
          </React.Suspense>
        </section>

        <section className="leaderboard-section" id="leaderboard" aria-labelledby="leaderboard-title">
          <div className="section-heading leaderboard-heading">
            <div>
              <h2 id="leaderboard-title">{t.leaderboard.title}</h2>
            </div>
            <div className="leaderboard-intro">
              <a className="leaderboard-contact" href={`mailto:${CONTACT_EMAIL}`}>
                <span className="leaderboard-contact-label">
                  <Mail size={18} aria-hidden="true" />
                  {t.leaderboard.contact}
                </span>
                <span className="leaderboard-contact-email">{CONTACT_EMAIL}</span>
                <ArrowUpRight className="leaderboard-contact-arrow" size={17} aria-hidden="true" />
              </a>
            </div>
          </div>

          <div className="leaderboard-tools">
            <div className="segmented-control leaderboard-version-control" role="tablist" aria-label={t.leaderboard.versionLabel}>
              {Object.entries(t.leaderboard.versions).map(([value, label]) => (
                <button
                  type="button"
                  role="tab"
                  key={value}
                  id={`leaderboard-version-tab-${value}`}
                  aria-selected={leaderboardVersion === value}
                  aria-controls="leaderboard-table"
                  className={leaderboardVersion === value ? "is-selected" : ""}
                  onClick={() => setLeaderboardVersion(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="tool-spacer" />
            <div className="segmented-control leaderboard-filter-control" aria-label={t.leaderboard.filterLabel}>
              {[
                ["all", t.leaderboard.filters.all],
                ["proprietary", t.leaderboard.filters.proprietary],
                ["open", t.leaderboard.filters.open],
              ].map(([value, label]) => (
                <button
                  type="button"
                  key={value}
                  className={modelType === value ? "is-selected" : ""}
                  onClick={() => setModelType(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <label className="select-control">
              <span className="sr-only">{t.leaderboard.domainLabel}</span>
              <select value={domain} onChange={(event) => setDomain(event.target.value)}>
                {Object.entries(t.domains).map(([value, label]) => (
                  <option key={value} value={value}>{domainDisplayName(value, label)}</option>
                ))}
              </select>
              <ChevronDown size={15} aria-hidden="true" />
            </label>
          </div>

          <div
            className="table-wrap"
            id="leaderboard-table"
            role="tabpanel"
            aria-labelledby={`leaderboard-version-tab-${leaderboardVersion}`}
            aria-live="polite"
          >
            <table>
              <colgroup>
                <col className="leaderboard-col-rank" />
                <col className="leaderboard-col-model" />
                <col className="leaderboard-col-harness" />
                <col className="leaderboard-col-score" />
                <col className="leaderboard-col-cost" />
                <col className="leaderboard-col-org" />
                <col className="leaderboard-col-date" />
              </colgroup>
              <thead>
                <tr>
                  <th scope="col">{t.leaderboard.columns.rank}</th>
                  <th scope="col">{t.leaderboard.columns.model}</th>
                  <th scope="col">{t.leaderboard.columns.harness}</th>
                  <th scope="col" className="score-column">{domainDisplayName(domain, t.domains[domain])} {t.leaderboard.columns.score}</th>
                  <th scope="col">{t.leaderboard.columns.cost}</th>
                  <th scope="col">{t.leaderboard.columns.org}</th>
                  <th scope="col">{t.leaderboard.columns.date}</th>
                </tr>
              </thead>
              <tbody>
                {filteredModels.length > 0 ? filteredModels.map((item, index) => {
                    const score = item.scores[domain];
                    const isRanked = item.ranked !== false;
                    const rank = isRanked ? index + 1 : null;
                    return (
                      <tr key={item.model}>
                        <td>
                          <span className={isRanked ? `rank rank-${rank}` : "rank rank-unranked"}>
                            {isRanked ? String(rank).padStart(2, "0") : "-"}
                          </span>
                        </td>
                        <td>
                          <strong className="model-name">{item.model}</strong>
                          {item.note && <span className="model-note">{item.note}</span>}
                        </td>
                        <td><span className="harness-name">{item.harness}</span></td>
                        <td className="score-cell">
                          <div className="score-number">{score.toFixed(2)}%</div>
                          <div className="score-track" aria-hidden="true">
                            <span style={{ width: `${(score / maxScore) * 100}%` }} />
                          </div>
                        </td>
                        <td className="cost-cell">
                          <span className="cost-total">{item.cost}</span>
                          {item.costPerTask && <span className="cost-per-task">{item.costPerTask}</span>}
                          {item.costUnit && <span className="cost-unit">{item.costUnit}</span>}
                        </td>
                        <td className="org-cell">{item.org}</td>
                        <td className="date-cell">{item.date}</td>
                      </tr>
                    );
                  }) : (
                    <tr>
                      <td className="leaderboard-empty" colSpan={7}>{t.leaderboard.emptyState}</td>
                    </tr>
                  )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="benchmark-section" id="benchmark" aria-labelledby="benchmark-title">
          <div className="section-heading benchmark-heading">
            <h2 id="benchmark-title">{t.benchmark.title}</h2>
          </div>

          <div className="benchmark-analysis">
            <figure className="domain-figure">
              <div className="domain-figure-media">
                <img src={domainDistributionFigure} alt={t.hero.figureCaption} />
              </div>
            </figure>

            <div className="pattern-grid">
              {t.benchmark.patterns.map(([title, description]) => (
                <article className="pattern-item" key={title}>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="findings-section" id="findings" aria-labelledby="findings-title">
          <div className="finding-lead">
            <h2 id="findings-title">{t.findings.title}</h2>
          </div>
          <figure className="performance-figure">
            <div className="performance-figure-media">
              <img src={performanceDegradationFigure} alt={t.findings.figureCaption} />
            </div>
            <figcaption>{t.findings.figureCaption}</figcaption>
          </figure>
          <div className="finding-metrics">
            {t.findings.metrics.map(([value, title, description]) => (
              <article key={title}>
                <span className="metric-number">{value}</span>
                <h3>{title}</h3>
                <p>{description}</p>
              </article>
            ))}
          </div>
        </section>

        <QuickStart content={t.quickStart} />

      </main>

      <footer>
        <a className="wordmark footer-mark" href="#top"><span className="wordmark-mark">L<span>↺</span></span><span>LongDS</span></a>
        <span className={visitorCount !== null ? "visit-counter is-ready" : "visit-counter"} aria-hidden={visitorCount === null}>
          <span>{t.views}</span>
          <strong>{visitorCount?.toLocaleString(language === "zh" ? "zh-CN" : "en-US")}</strong>
        </span>
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
