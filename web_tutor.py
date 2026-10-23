# web_tutor.py —— AI校园（v3.3：LINE黑白极简UI · 手机优先 · 旧档案兼容修复）
# 运行方式：streamlit run web_tutor.py
# 建议 streamlit >= 1.39（pip install -U streamlit）

import os
import re
import json
import time
import math
import random
import base64
import difflib
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from teachers import (
    TEACHERS, CHARACTER_POOL, build_classmate_prompt,
    SUMMON_COST, REWARD_CORRECT, REWARD_WRONG, DUPLICATE_REFUND, DROP_WEIGHTS,
)

try:
    import sympy
    SYMPY_OK = True
except ImportError:
    SYMPY_OK = False

load_dotenv()

# ================= 配置 =================
TEACHER_MODEL = "glm-4-flash"
VISION_MODEL = "glm-4v-flash"
SOLVER_MODEL = "glm-4.5-flash"
ASR_MODEL = "glm-asr-2512"
TEACHER_TEMPERATURE = 0.3
WRONG_BOOK_FILE = "错题本.json"
LAST_TIME_FILE = "上次学习时间.txt"
PROFILE_FILE = "学生档案.json"
WALLET_FILE = "钱包.json"
UNLOCKED_FILE = "已解锁角色.json"
DYNAMICS_FILE = "学习动态.json"
UNREAD_FILE = "同学消息.json"
REWARD_CLASSMATE = 5
AUTO_REFRESH_SEC = 45              # 仅旧版streamlit的回退刷新用


def _st_ver():
    try:
        parts = re.findall(r"\d+", st.__version__ or "")[:2]
        return tuple(int(p) for p in parts)
    except Exception:
        return (0, 0)


# fragment + st.rerun(scope="app") 需要 streamlit >= 1.39
SUPPORTS_FRAGMENT = hasattr(st, "fragment") and _st_ver() >= (1, 39)

# ================= 业务数据 =================
KNOWLEDGE = {
    "射影定理": "直角三角形中，斜边上的高是两条直角边在斜边上射影的比例中项。"
             "p、q 是【斜边被高分成的两段】。公式：h² = p·q；a² = p·c，b² = q·c。"
             "已知两直角边 a、b 和斜边 c 时，高用面积法：h = a×b÷c。",
    "勾股定理": "直角三角形两直角边 a、b 与斜边 c 满足：a² + b² = c²。",
    "勾股定理逆定理": "若三边满足 a² + b² = c²，则该三角形是直角三角形。",
    "圆柱体积": "圆柱体积 = 底面积 × 高，V = πr²h。",
    "圆的面积": "圆的面积 S = πr²，周长 C = 2πr。",
    "三角形全等": "判定：SSS、SAS、ASA、AAS、HL。",
    "相似三角形": "对应角相等、对应边成比例。判定：AA；两边成比例且夹角相等；三边成比例。",
    "牛顿第二定律": "F = ma。",
    "密度公式": "ρ = m/V。",
    "压强公式": "p = F/S；液体内部压强 p = ρgh。",
    "欧姆定律": "I = U/R。",
    "速度公式": "v = s/t。",
}

CHALLENGE_WORDS = ("不对", "错了", "是不是错", "有问题", "质疑", "确定吗",
                   "真的吗", "应该不是", "搞错", "不太对", "好像不对")
ASSERT_WORDS = ("正确答案", "答案应该", "应该是", "应为")
HELP_WORDS = ("不知道", "不会", "不清楚", "忘了", "没学过", "教我",
              "不明白", "不懂", "听不懂", "再讲", "讲解", "检查", "继续")
NEGATION_WORDS = ("不是", "不等于", "并非", "≠")
QUESTION_HINTS = ("什么", "怎么", "如何", "为什么", "为何", "介绍", "讲讲", "？", "?")

PYTHAGOREAN_TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13),
                       (9, 12, 15), (8, 15, 17), (7, 24, 25), (20, 21, 29)]

TEACHING_RULES = """

【教学立场规则（最高优先级）】
1. 一切答案以系统【标准答案】为唯一依据（程序验证），禁止自己另算。
2. 学生质疑时不得改口，一步步展示推导引导学生自己发现问题。
3. 严禁编造定理公式。4. 不用讨好式问句。5. 权威来自正确，不来自顺从。
"""

STAGE_OPENING = "（你正在写作业，同学凑了过来向你请教。记住：用户是Ta的同桌，你是{name}。）"


# ================= 持久化小工具 =================
def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_wallet():
    return _load_json(WALLET_FILE, {"积分": 200, "累计获得": 0})


def save_wallet(w):
    _save_json(WALLET_FILE, w)


def add_points(amount):
    w = load_wallet()
    w["积分"] += amount
    if amount > 0:
        w["累计获得"] += amount
    save_wallet(w)
    return w["积分"]


def load_unlocked():
    return _load_json(UNLOCKED_FILE, ["c_xiaohang"])


def save_unlocked(lst):
    _save_json(UNLOCKED_FILE, lst)


def load_profile():
    profile = _load_json(PROFILE_FILE, {"年级": "", "最近学习": [], "掌握": [], "薄弱": []})
    # v3.3 兼容早期版本的档案文件：缺失的键自动补齐，避免 KeyError
    profile.setdefault("年级", "")
    profile.setdefault("最近学习", [])
    profile.setdefault("掌握", [])
    profile.setdefault("薄弱", [])
    return profile


def save_profile(p):
    _save_json(PROFILE_FILE, p)


def add_study_topic(topic):
    profile = load_profile()
    profile.setdefault("最近学习", [])
    topic = topic.strip()
    if topic and topic not in profile["最近学习"]:
        profile["最近学习"].insert(0, topic)
        profile["最近学习"] = profile["最近学习"][:12]
        save_profile(profile)


def update_mastery(topic, good=True):
    profile = load_profile()
    # v3.3 兼容旧档案：缺失的键自动补上
    profile.setdefault("年级", "")
    profile.setdefault("最近学习", [])
    profile.setdefault("掌握", [])
    profile.setdefault("薄弱", [])
    key = "掌握" if good else "薄弱"
    other = "薄弱" if good else "掌握"
    topic = topic.strip()
    if topic and topic not in profile[key]:
        profile[key].insert(0, topic)
        profile[key] = profile[key][:8]
    if topic in profile[other]:
        profile[other].remove(topic)
    save_profile(profile)


def profile_card():
    profile = load_profile()
    lines = []
    if profile.get("年级"):
        lines.append(f"用户年级：{profile['年级']}")
    if profile.get("最近学习"):
        lines.append("最近学习内容（新→旧）：")
        for item in profile["最近学习"]:
            lines.append(f"- {item}")
    if profile.get("掌握"):
        lines.append("已确认掌握（同学问答验证）：")
        for item in profile["掌握"]:
            lines.append(f"- {item}")
    if profile.get("薄弱"):
        lines.append("存在薄弱（同学问答暴露，请优先巩固）：")
        for item in profile["薄弱"]:
            lines.append(f"- {item}")
    return "\n".join(lines)


# ================= 同学提问队列 =================
def pick_classmate():
    unlocked = load_unlocked()
    return random.choice(unlocked) if unlocked else None


def schedule_classmate_question(topic, excerpt):
    """③ 老师对话结束后必定调用：随机延迟安排同学提问"""
    cid = pick_classmate()
    if not cid or not topic:
        return
    entries = _load_json(DYNAMICS_FILE, [])
    for e in entries[-10:]:
        if e["topic"] == topic and e["status"] in ("pending", "asked"):
            return
    entries.append({
        "topic": topic,
        "excerpt": excerpt[:600],
        "classmate": cid,
        "trigger": time.time() + random.randint(30, 120),
        "status": "pending",
        "question": "", "ref_answer": "",
        "attempts": 0,
    })
    _save_json(DYNAMICS_FILE, entries[-30:])


def generate_classmate_question(entry):
    char = next((c for c in CHARACTER_POOL if c["id"] == entry["classmate"]), None)
    if not char:
        return None
    knowledge_hint = ""
    for name, desc in KNOWLEDGE.items():
        if name in entry["topic"] or name in entry["excerpt"]:
            knowledge_hint = f"【知识点标准表述·仅参考】\n{desc}"
            break
    prompt = (
        f"你是{char['name']}（{char['identity']}）。{char['personality']}\n"
        f"你刚听了同桌和老师讨论的内容，想考考TA（用你的语气，像发微信）。\n"
        f"【讨论主题】{entry['topic']}\n"
        f"【老师讲解摘录】{entry['excerpt']}\n"
        f"{knowledge_hint}\n\n"
        "任务：围绕同一知识点出一道小问题，难度比讨论内容稍简单或稍难一点"
        "（二选一），必须是具体、可以简短回答的知识性问题（含计算或概念辨析）。\n"
        '只输出JSON：{"question": "你发给同桌的问题（用你的人设语气，最多3句话）", '
        '"ref_answer": "该问题的正确答案（简短，标准表述，供系统判卷用）"}\n'
        "JSON外不要有任何文字。"
    )
    try:
        raw = call_model(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if data.get("question") and data.get("ref_answer"):
                return data
    except Exception:
        pass
    return None


def process_due_questions():
    """把到期的提问生成为未读消息。② 生成失败改为重试（最多3次），不再直接丢弃"""
    entries = _load_json(DYNAMICS_FILE, [])
    changed = False
    now = time.time()
    for e in entries:
        if e["status"] == "pending" and e["trigger"] <= now:
            result = generate_classmate_question(e)
            if result:
                unread = _load_json(UNREAD_FILE, {})
                unread.setdefault(e["classmate"], []).append({
                    "type": "quiz", "topic": e["topic"],
                    "text": result["question"], "time": time.strftime("%H:%M"),
                })
                _save_json(UNREAD_FILE, unread)
                e["question"] = result["question"]
                e["ref_answer"] = result["ref_answer"]
                e["status"] = "asked"
            else:
                # ② 重试机制：最多3次，每次延后60秒
                e["attempts"] = e.get("attempts", 0) + 1
                if e["attempts"] >= 3:
                    e["status"] = "failed"
                else:
                    e["trigger"] = time.time() + 60
            changed = True
    if changed:
        _save_json(DYNAMICS_FILE, entries)


def get_unread(cid):
    return _load_json(UNREAD_FILE, {}).get(cid, [])


def clear_unread(cid):
    unread = _load_json(UNREAD_FILE, {})
    unread[cid] = []
    _save_json(UNREAD_FILE, unread)


def get_pending_quiz(cid):
    entries = _load_json(DYNAMICS_FILE, [])
    for e in reversed(entries):
        if e["classmate"] == cid and e["status"] == "asked":
            return e
    return None


def mark_quiz_status(cid, question_text, status):
    """⑤ 正确地把提问状态写回文件（修复原版内存改动被覆盖的bug）"""
    entries = _load_json(DYNAMICS_FILE, [])
    for e in reversed(entries):
        if (e.get("classmate") == cid and e.get("question") == question_text
                and e.get("status") == "asked"):
            e["status"] = status
            break
    _save_json(DYNAMICS_FILE, entries)


def resolve_quiz(entry, user_answer):
    char = next((c for c in CHARACTER_POOL if c["id"] == entry["classmate"]), None)
    persona = f"{char['name']}（{char['identity']}）。{char['personality']}" if char else "同学"
    prompt = (
        f"你是{persona}\n"
        f"你问了同桌一个问题：【{entry['question']}】\n"
        f"【正确答案（系统判定依据）】{entry['ref_answer']}\n"
        f"【同桌的回答】{user_answer}\n\n"
        "先严格判定对错（意思对、数值等价即算对）。"
        "如果这条回答是在回应你上一轮的追问，只要他说出了正确思路就算对。\n"
        "若答对：用你的人设表达『听明白了/牛啊/原来是这样』之类的恍然大悟（1-2句话）。\n"
        "若答错：不要直接说答案！用你的困惑暴露他错的地方，"
        "追问一个引导他重新想的反问（1-2句话）。\n"
        '只输出JSON：{"correct": true或false, "reply": "你发给同桌的回复"}\n'
        "JSON外不要有任何文字。"
    )
    try:
        raw = call_model(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return bool(data.get("correct")), data.get("reply", "")
    except Exception:
        pass
    return None, "（网络出错，稍后再试）"


# ================= LaTeX 清洗 =================
def clean_formula(text):
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "").replace("$", "")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_([a-zA-Z0-9])", r"\1", text)
    for old, new in {
        "\\times": "×", "\\cdot": "·", "\\pi": "π", "\\approx": "≈",
        "\\pm": "±", "\\leq": "≤", "\\geq": "≥", "\\neq": "≠",
        "\\Rightarrow": "→", "\\rightarrow": "→", "\\left": "", "\\right": "",
        "\\alpha": "α", "\\beta": "β", "\\theta": "θ", "\\Delta": "Δ",
    }.items():
        text = text.replace(old, new)
    text = re.sub(r"\^\{?2\}?", "²", text)
    text = re.sub(r"\^\{?3\}?", "³", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return text


def _braces_closed(text, count):
    depth = 0
    closed = 0
    for char in text:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                closed += 1
                if closed == count:
                    return True
    return False


def is_mature(chunk):
    if len(chunk) > 80:
        return True
    if chunk.startswith("\\"):
        match = re.match(r"\\([a-zA-Z]+)", chunk)
        if not match:
            return len(chunk) >= 2
        command = match.group(1)
        rest = chunk[1 + len(command):]
        if command in ("frac", "sqrt", "text", "mathrm"):
            need = 2 if command == "frac" else 1
            return bool(rest) and _braces_closed(rest, need)
        return len(rest) >= 1 and not rest[0].isalpha()
    if chunk.startswith("**"):
        return chunk.count("**") >= 2
    if chunk.startswith("*"):
        return len(chunk) >= 2
    if chunk.startswith("^") or chunk.startswith("_"):
        if len(chunk) < 2:
            return False
        if chunk[1] == "{":
            return _braces_closed(chunk[1:], 1)
        return True
    if chunk.startswith("$"):
        return len(chunk) >= 2
    return True


def extract_printable(pending):
    positions = [pending.find(c) for c in ("\\", "*", "^", "$", "_") if c in pending]
    if not positions:
        return pending, ""
    start = min(positions)
    safe, chunk = pending[:start], pending[start:]
    if is_mature(chunk):
        return safe + clean_formula(chunk), ""
    return safe, chunk


# ================= 模型调用 =================
@st.cache_resource
def get_client():
    return OpenAI(
        api_key=os.getenv("GLM_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )


client = get_client()


def call_model(prompt, models=(SOLVER_MODEL, TEACHER_MODEL)):
    last_error = None
    for model_name in models:
        try:
            resp = client.chat.completions.create(
                model=model_name, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as error:
            last_error = error
    raise last_error


def transcribe_audio(audio_bytes, filename="voice.wav"):
    try:
        resp = client.audio.transcriptions.create(
            model=ASR_MODEL,
            file=(filename, audio_bytes, "audio/wav"),
        )
        return getattr(resp, "text", "").strip()
    except Exception:
        return None


def parse_json(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return None


def format_number(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


# ================= 答案锚点 =================
def solve_and_verify(question):
    if not SYMPY_OK:
        return None
    prompt = (
        "解下面这道初中数学题：先想清楚求什么，再写Python代码计算。\n"
        f"【题目】{question}\n"
        "只输出一个Python代码块（python 开头），要求：\n"
        "1. 第一行 from sympy import *\n"
        "2. 最终答案存入变量 answer（能精确就精确）\n"
        "3. 不要 input()，不要 print，不要解释文字"
    )
    try:
        raw = call_model(prompt)
    except Exception:
        return None
    FENCE = "`" * 3
    match = re.search(FENCE + r"(?:python)?\s*(.*?)" + FENCE, raw, re.DOTALL)
    code = match.group(1) if match else raw
    if ("answer" not in code) or ("input(" in code):
        return None
    try:
        namespace = {}
        exec(code, namespace)  # 本地自用：代码由解题模型生成，已过滤 input
        answer_obj = namespace.get("answer")
        if answer_obj is None:
            return None
        try:
            value = float(answer_obj)
        except Exception:
            value = None
        answer_text = format_number(value) if value is not None else str(answer_obj)
        return {"answer_text": answer_text, "value": value}
    except Exception:
        return None


def lock_question(teacher_reply):
    try:
        raw = call_model(
            "下面是一位老师发给学生的最后一条消息。判断结尾是否向学生提出了需要给出具体知识性答案的问题。\n"
            f"【老师消息】{teacher_reply}\n"
            '只输出JSON：{"is_question": true或false, "type": "knowledge"或"process", "question": "题面原文"}\n'
            "过程性/确认性问题→process；需要知识性答案→knowledge，并把题面重写为独立可解的完整题目。"
            "JSON外不要有任何文字。"
        )
        data = parse_json(raw)
        if not data or not (data.get("is_question") and data.get("type") == "knowledge"):
            return None
        question = (data.get("question") or "").strip()
        if not question:
            return None
        result = solve_and_verify(question)
        if result:
            return {"question": question, "answer": result["answer_text"],
                    "value": result["value"], "verified": True}
    except Exception:
        pass
    return None


# ================= 判卷 =================
def extract_numbers(text):
    normalized = (text.replace("，", " ").replace("。", " ")
                  .replace("×", "*").replace("÷", "/").replace("−", "-"))
    values = []
    for expr in re.findall(r"-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+", normalized):
        if re.fullmatch(r"[-\d.\s*/+]+", expr):
            try:
                values.append(float(eval(expr, {"__builtins__": {}}, {})))
            except Exception:
                pass
    for match in re.findall(
        r"(-?\d+(?:\.\d+)?)?\s*(?:\*|×)?\s*倍?\s*(?:√|根号)\s*\(?\s*(\d+(?:\.\d+)?)", normalized,
    ):
        try:
            coeff = float(match[0]) if match[0] else 1.0
            values.append(coeff * math.sqrt(float(match[1])))
        except Exception:
            pass
    for num in re.findall(r"-?\d+(?:\.\d+)?", normalized):
        values.append(float(num))
    return values


def numbers_close(a, b):
    return abs(a - b) < 1e-9 or round(a, 2) == round(b, 2)


def grade_answer(anchor, student_answer):
    if anchor.get("value") is not None and not any(w in student_answer for w in NEGATION_WORDS):
        values = extract_numbers(student_answer)
        if values:
            if any(numbers_close(v, anchor["value"]) for v in values):
                return True, f"回答正确（标准答案 {anchor['answer']}）"
            return False, f"与标准答案 {anchor['answer']} 不符"
    try:
        raw = call_model(
            "你是严格的阅卷老师。\n"
            f"【题目】{anchor['question']}\n【标准答案（程序验证）】{anchor['answer']}\n"
            f"【学生回答】{student_answer}\n"
            '只输出JSON：{"correct": true或false或"skip", "note": "20字内点评"}。'
            "一致→true，不一致→false，没在答题→skip。数值等价即一致。JSON外不要有任何文字。"
        )
        data = parse_json(raw)
        if not data:
            return None, ""
        if data.get("correct") == "skip":
            return "skip", data.get("note", "学生切换了话题")
        return bool(data.get("correct")), data.get("note", "")
    except Exception:
        return None, ""


# ================= 出题 =================
def generate_variation():
    a, b, c = random.choice(PYTHAGOREAN_TRIPLES)
    if random.random() < 0.5:
        return f"直角三角形的两条直角边分别为 {a} 和 {b}，斜边为 {c}。求斜边上的高 h。", a * b / c
    return f"直角三角形的两条直角边分别为 {a} 和 {b}，斜边为 {c}。求直角边 {a} 在斜边上的射影长 p。", a * a / c


def make_anchor(text, value):
    return {"question": text, "answer": format_number(value), "value": value, "verified": True}


# ================= 其他工具 =================
def match_knowledge(text):
    if not text:
        return None
    best, best_ratio = None, 0.0
    for name in KNOWLEDGE:
        if name in text:
            return name
        size = len(name)
        for i in range(max(0, len(text) - size + 1)):
            ratio = difflib.SequenceMatcher(None, text[i:i + size], name).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, name
    return best if best_ratio >= 0.7 else None


def save_wrong_question(anchor, student_answer, note):
    records = _load_json(WRONG_BOOK_FILE, [])
    records.append({
        "时间": time.strftime("%Y-%m-%d %H:%M"),
        "题目": anchor["question"][:300],
        "标准答案": anchor.get("answer", ""),
        "学生回答": student_answer,
        "点评": note,
    })
    _save_json(WRONG_BOOK_FILE, records)


def is_challenge(text):
    return any(word in text for word in CHALLENGE_WORDS)


# ================= 联系人注册表 =================
def build_contacts():
    contacts = {}
    for key, t in TEACHERS.items():
        contacts[key] = {**t, "kind": "teacher", "no_judge": False}
    unlocked = load_unlocked()
    for c in CHARACTER_POOL:
        if c["id"] in unlocked:
            contacts[c["id"]] = {
                "subject": "数学", "name": c["name"], "avatar": c["avatar"],
                "kind": "classmate", "no_judge": True,
                "prompt": build_classmate_prompt(c), "stars": c["stars"],
            }
    return contacts


# ================= LINE 风格 UI（黑白极简 · 手机优先） =================
st.set_page_config(page_title="AI校园", page_icon="🏫", layout="centered")

st.markdown("""
<style>
/* ---------- 全局基调 ---------- */
.stApp { background: #ffffff; }
.stApp, .stApp button, .stApp input, .stApp textarea {
    font-family: -apple-system, "PingFang SC", "Helvetica Neue",
                 "Microsoft YaHei", "Segoe UI", sans-serif;
}
#MainMenu, footer, [data-testid="stFooter"], [data-testid="stToolbar"] {
    display: none !important;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

/* ---------- 标题（LINE黑条） ---------- */
h1 {
    font-size: 1.05rem !important; font-weight: 700; text-align: center;
    color: #ffffff !important; background: #000000;
    padding: 12px 0 !important; margin: 0 0 6px 0 !important;
    letter-spacing: 1px;
}

/* ---------- 侧边栏（白底细线列表） ---------- */
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #eeeeee; }
[data-testid="stSidebar"] * { color: #111111 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-size: 0.95rem !important; font-weight: 700; }
[data-testid="stSidebar"] hr { border-color: #f0f0f0; margin: 8px 0 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 10px 8px !important; border-bottom: 1px solid #f4f4f4;
    border-radius: 0 !important; gap: 10px;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #f8f8f8; }

/* ---------- 主区域：手机宽度居中 ---------- */
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 560px; margin: 0 auto;
    padding: 8px 12px 110px 12px !important;
}

/* ---------- 聊天气泡（LINE风） ---------- */
[data-testid="stChatMessage"] {
    background: transparent; box-shadow: none !important; border: none;
    padding: 4px 0 !important; max-width: 90%;
}
/* 对方：白底灰边气泡，靠左 */
[data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) {
    align-self: flex-start;
}
[data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"]))
[data-testid="stChatMessageContent"] {
    background: #ffffff; border: 1px solid #e5e5e5;
    border-radius: 4px 18px 18px 18px;
    padding: 10px 14px; color: #111111;
    font-size: 0.95rem; line-height: 1.55;
}
/* 自己：黑色气泡白字，靠右，隐藏自己头像（LINE习惯） */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse; margin-left: auto;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="stChatMessageContent"] {
    background: #111111; color: #ffffff;
    border-radius: 18px 4px 18px 18px;
    padding: 10px 14px; font-size: 0.95rem; line-height: 1.55;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="chatAvatarIcon-user"] { display: none; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="stChatMessageContent"] a { color: #9fe870; }

/* 系统提示 → 细灰条（判卷结果、积分提示都不抢戏） */
[data-testid="stAlert"] {
    background: #fafafa !important; color: #777777 !important;
    border: none !important; border-left: 3px solid #dddddd !important;
    padding: 8px 12px !important; font-size: 0.82rem !important;
}

/* ---------- 输入区 ---------- */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] > div {
    border: 1px solid #dddddd !important; border-radius: 22px !important;
}
[data-testid="stChatInput"] textarea { font-size: 0.95rem; }
[data-testid="stChatInput"] button {
    background: #111111 !important; border-radius: 50% !important;
    color: #ffffff !important;
}

/* 按钮：黑描边 / 黑底 */
.stButton > button {
    border-radius: 10px; font-weight: 600;
    border: 1px solid #111111; background: #ffffff; color: #111111;
}
.stButton > button:hover { background: #111111; color: #ffffff; border-color: #111111; }
.stButton > button[kind="primary"] { background: #111111; color: #ffffff; }

/* 上传/录音弱化成细线小件 */
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #dddddd !important; border-radius: 10px;
    padding: 6px 10px; font-size: 0.8rem; color: #888888;
    background: #ffffff;
}

/* 折叠面板（输入工具 / 错题本） */
[data-testid="stExpander"] { border: 1px solid #eeeeee !important; border-radius: 12px !important; }
[data-testid="stExpander"] summary { font-size: 0.85rem; color: #666666; }
[data-testid="stExpander"] p { font-size: 0.9rem; }

/* toast：黑色胶囊 */
[data-testid="stToast"] { background: #111111; color: #ffffff; border-radius: 22px; }

/* 手机端适配 */
@media (max-width: 640px) {
    .block-container { padding-left: 10px !important; padding-right: 10px !important; }
    [data-testid="stSidebar"] { min-width: 78vw !important; }
    h1 { font-size: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)

if not SYMPY_OK:
    st.error("未安装 sympy，答案自动验证不可用！请先运行：pip install sympy")
    st.stop()

# 每次页面运行先处理到期的同学提问（fragment里也会调用，幂等）
process_due_questions()

wallet = load_wallet()

# ================= 侧边栏 =================
with st.sidebar:
    view = st.radio("导航", ["💬 聊天", "👥 联系人", "🎴 召唤"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("### 💰 积分钱包")
    st.markdown(f"**{wallet['积分']}** 分　（累计获得 {wallet['累计获得']}）")
    st.caption("答对题目获得积分，积分可召唤新同学")

    contacts = build_contacts()
    total_unread = sum(len(v) for v in _load_json(UNREAD_FILE, {}).values())
    if total_unread:
        st.error(f"🔔 你有 {total_unread} 条未读消息！")

    if view == "💬 聊天":
        st.divider()
        st.header("👤 选择联系人")

        def contact_label(k):
            c = contacts[k]
            badge = ""
            n = len(get_unread(k))
            if n:
                badge = f" 🔴{n}"
            star = " ★" * c.get("stars", 0) if c["kind"] == "classmate" else ""
            return f"{c['avatar']} {c['name']}{star}{badge}"

        choice = st.radio("联系人", list(contacts.keys()),
                          format_func=contact_label, label_visibility="collapsed")
        profile = load_profile()
        st.divider()
        st.caption("🪪 我的身份（档案）")
        new_grade = st.text_input("年级", value=profile.get("年级", ""),
                                  placeholder="如：初三 / 高一", label_visibility="collapsed")
        if new_grade != profile.get("年级", ""):
            profile["年级"] = new_grade.strip()
            save_profile(profile)
        web_search_on = st.checkbox("🌐 联网搜索（老师）", value=True)
        auto_refresh = st.checkbox("🔔 实时消息提醒", value=True,
                                   help="后台每15秒静默检查，有新消息才刷新页面，不打断输入"
                                        "（需 streamlit ≥ 1.39）")
        st.divider()
        save_clicked = st.button("💾 保存并结束本次对话", use_container_width=True)

# ⑦ 浏览器标签页标题提醒（有未读时提示；无需系统授权）
if view == "💬 聊天":
    _tab_title = "🔔 新消息 · AI校园" if total_unread else "AI校园"
    st.html(f"<script>document.title={_tab_title!r};</script>")

# ================= 召唤页 =================
if view == "🎴 召唤":
    st.title("🎴 召唤新同学")
    st.caption("用学习积分召唤新的同学角色。星级越高，性格与人设越丰富；"
               "所有同学的学习功能完全一致。")

    if "summon_result" in st.session_state:
        kind, msg = st.session_state.pop("summon_result")
        if kind == "new":
            st.balloons()
        (st.error if kind == "fail" else st.success)(msg)

    unlocked = load_unlocked()
    wallet_now = load_wallet()

    cols = st.columns(3)
    for i, c in enumerate(CHARACTER_POOL):
        with cols[i % 3]:
            owned = c["id"] in unlocked
            st.markdown(f"**{c['avatar']} {c['name']}**\n\n{'⭐' * c['stars']}\n\n{c['identity']}")
            st.caption(c["personality"][:40] + "…")
            if owned:
                st.success("已拥有")
            else:
                if st.button(f"召唤（{SUMMON_COST}分）", key=f"gacha_{c['id']}"):
                    st.session_state.pending_summon_target = c["id"]
                    st.rerun()

    st.divider()
    if st.button(f"🎴 随机召唤一次（消耗 {SUMMON_COST} 积分）", use_container_width=True,
                 disabled=wallet_now["积分"] < SUMMON_COST):
        st.session_state.pending_summon_random = True
        st.rerun()
    if wallet_now["积分"] < SUMMON_COST:
        st.warning("积分不足——去和老师答题赚积分吧！")

    target = st.session_state.pop("pending_summon_target", None)
    do_random = st.session_state.pop("pending_summon_random", False)
    if target or do_random:
        w = load_wallet()
        if w["积分"] >= SUMMON_COST:
            w["积分"] -= SUMMON_COST
            save_wallet(w)
            if target:
                result = next(c for c in CHARACTER_POOL if c["id"] == target)
            else:
                stars = random.choices(list(DROP_WEIGHTS.keys()),
                                       weights=list(DROP_WEIGHTS.values()))[0]
                pool = [c for c in CHARACTER_POOL if c["stars"] == stars]
                result = random.choice(pool)
            unlocked_now = load_unlocked()
            if result["id"] in unlocked_now:
                add_points(DUPLICATE_REFUND)
                st.session_state.summon_result = (
                    "dup", f"已经是老同学了！转化为默契值 +{DUPLICATE_REFUND} 积分")
            else:
                unlocked_now.append(result["id"])
                save_unlocked(unlocked_now)
                st.session_state.summon_result = (
                    "new", f"🎉 召唤成功：{result['avatar']} {result['name']}"
                           f"（{'⭐' * result['stars']}）已加入你的联系人！")
        else:
            st.session_state.summon_result = ("fail", "积分不足，召唤失败（未扣分）")
        st.rerun()

# ================= 联系人页 =================
elif view == "👥 联系人":
    st.title("👥 联系人")
    unlocked = load_unlocked()
    st.markdown("--- **老师** ---")
    for key, t in TEACHERS.items():
        st.markdown(f"{t['avatar']} **{t['name']}**　({t['subject']} · 一对一辅导)")
    st.markdown("--- **同学（学完老师课后会主动来找你）** ---")
    for c in CHARACTER_POOL:
        if c["id"] in unlocked:
            n = len(get_unread(c["id"]))
            badge = f"　🔴 {n}条新消息" if n else ""
            st.markdown(f"{c['avatar']} **{c['name']}**　{'⭐' * c['stars']}　— {c['identity']}{badge}")
        else:
            st.markdown(f"🔒 **？？？**　{'⭐' * c['stars']}　— 前往「🎴 召唤」解锁")

# ================= 聊天页 =================
else:
    teacher = contacts[choice]
    no_judge = teacher["no_judge"]
    assistant_avatar = teacher["avatar"]
    cid = choice
    st.title(f"💬 {teacher['name']}")

    # ---------- ① 静默轮询（fragment）：每15秒后台查未读，有新消息才整页刷新 ----------
    if auto_refresh:
        if SUPPORTS_FRAGMENT:
            @st.fragment(run_every=15)
            def poll_unread():
                process_due_questions()   # 到期的提问在后台静默生成
                total = sum(len(v) for v in _load_json(UNREAD_FILE, {}).values())
                if total > st.session_state.get("unread_baseline", 0):
                    st.session_state.unread_baseline = total
                    st.toast("🔔 收到新消息")
                    st.rerun(scope="app")
            poll_unread()
        else:
            # 旧版回退：整页定时刷新（体验差，建议升级 streamlit）
            st.caption("⚠️ streamlit 版本较低，实时提醒退化为整页刷新。建议运行：pip install -U streamlit")
            st.markdown(f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SEC}">',
                        unsafe_allow_html=True)

    # ---------- 会话初始化 ----------
    if "contact_key" not in st.session_state or st.session_state.contact_key != cid:
        st.session_state.contact_key = cid
        base_prompt = teacher["prompt"] + ("" if no_judge else TEACHING_RULES)
        st.session_state.history = [{"role": "system", "content": base_prompt}]
        st.session_state.msgs = []
        st.session_state.anchor = None
        st.session_state.scores = []

        card = profile_card()
        if card:
            if no_judge:
                st.session_state.history.append({
                    "role": "system",
                    "content": (
                        f"【学生档案·仅你可见】\n{card}\n"
                        "重要：用户是你的同桌，不要搞混称呼。"
                        "『薄弱』列表里的内容是TA的弱点，提问时可优先参考。"
                    ),
                })
            else:
                st.session_state.history.append({
                    "role": "system",
                    "content": (
                        f"【学生档案·仅你可见】\n{card}\n"
                        "请据此调整讲解难度与举例：『薄弱』列表里的内容要重点巩固，"
                        "『掌握』列表里的可以快速带过或加大难度。"
                    ),
                })

        last_record = None
        if os.path.exists(LAST_TIME_FILE):
            with open(LAST_TIME_FILE, "r", encoding="utf-8") as f:
                last_record = f.read()
        gap_hint = ""
        if last_record:
            try:
                gap = datetime.now() - datetime.strptime(last_record, "%Y-%m-%d %H:%M")
                if gap.days >= 1:
                    gap_hint = f"对方已经{gap.days}天没来了，开场表达挂念，"
                elif gap.seconds // 3600 >= 2:
                    gap_hint = "对方几小时前来过，开场问一句是不是卡住了，"
            except Exception:
                pass
        with open(LAST_TIME_FILE, "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M"))
        if gap_hint:
            st.session_state.history.append({
                "role": "system",
                "content": f"【当前状态】{gap_hint}请在第一句话中自然体现。",
            })

        if no_judge:
            st.session_state.history.append({
                "role": "user",
                "content": STAGE_OPENING.format(name=teacher["name"]),
            })

    # ---------- 流式回复 ----------
    def stream_reply(messages, search_query=None):
        kwargs = dict(model=TEACHER_MODEL, messages=messages,
                      temperature=TEACHER_TEMPERATURE, stream=True)
        if search_query:
            kwargs["tools"] = [{
                "type": "web_search",
                "web_search": {"enable": True, "search_query": search_query,
                               "search_engine": "search_std", "count": 5},
            }]
        stream = client.chat.completions.create(**kwargs)
        chunk = ""
        for event in stream:
            if not event.choices:
                continue
            piece = event.choices[0].delta.content or ""
            if not piece:
                continue
            chunk += piece
            printable, chunk = extract_printable(chunk)
            if printable:
                yield printable
        if chunk:
            yield clean_formula(chunk)

    def reply(vision_image=None, vision_text="", search_query=None):
        if vision_image is not None:
            b64 = base64.b64encode(vision_image).decode()
            message = {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": vision_text or "（对方发来一张图片）"},
            ]}
            with st.chat_message("assistant", avatar=assistant_avatar):
                placeholder = st.empty()
                try:
                    resp = client.chat.completions.create(
                        model=VISION_MODEL, temperature=TEACHER_TEMPERATURE,
                        messages=st.session_state.history[:-1] + [message],
                    )
                    full = clean_formula(resp.choices[0].message.content or "")
                except Exception:
                    full = "（图片识别出错，稍后再试）"
                placeholder.write(full)
        else:
            with st.chat_message("assistant", avatar=assistant_avatar):
                full = st.write_stream(stream_reply(st.session_state.history, search_query))
        st.session_state.history.append({"role": "assistant", "content": full})
        st.session_state.msgs.append({"role": "assistant", "content": full})
        return full

    def sys_note(text, style="info"):
        st.session_state.msgs.append({"role": "note", "content": text, "style": style})
        {"ok": st.success, "err": st.error}.get(style, st.info)(text)

    def do_practice():
        question_text, answer_value = generate_variation()
        st.session_state.anchor = make_anchor(question_text, answer_value)
        sys_note("练习题已生成，标准答案已由程序锁定 🔒")
        st.session_state.history.append({
            "role": "system",
            "content": (f"【变式练习】请把这道题出给学生（可轻微改述，切勿透露答案）：\n{question_text}\n"
                        f"（系统已验证标准答案：{st.session_state.anchor['answer']}，仅你可见）"),
        })
        st.session_state.history.append({"role": "user", "content": "老师，请给我出一道练习题。"})
        st.session_state.msgs.append({"role": "user", "content": "老师，请给我出一道练习题。"})
        reply()
        sys_note("新题标准答案已锁定并经程序验证 ✅")

    # ---------- 保存 ----------
    if save_clicked:
        if st.session_state.msgs:
            filename = f"记录_{teacher['name']}_{time.strftime('%m%d_%H%M')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                for msg in st.session_state.history:
                    if msg["role"] == "system":
                        continue
                    if msg["role"] == "user" and msg["content"].startswith("（你正在写作业"):
                        continue
                    who = "学生" if msg["role"] == "user" else teacher["name"]
                    f.write(f"{who}：{msg['content']}\n\n")
            scores = st.session_state.scores
            if scores:
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(f"\n[本次成绩：{scores.count(True)}/{len(scores)} 题正确]\n")
            st.success(f"记录已保存：{filename}")
        else:
            st.warning("本次还没有对话。")
        st.stop()

    # ---------- ④ 渲染未读消息：先入对话历史，再rerun以正确排序 ----------
    unread_now = get_unread(cid)
    if unread_now:
        for u in unread_now:
            st.session_state.history.append({"role": "assistant", "content": u["text"]})
            st.session_state.msgs.append({"role": "assistant", "content": u["text"]})
        clear_unread(cid)
        # ④ 更新基线，防止fragment误判"又有新消息"
        st.session_state.unread_baseline = sum(
            len(v) for v in _load_json(UNREAD_FILE, {}).values())
        st.rerun()

    # ---------- 历史渲染 ----------
    for msg in st.session_state.msgs:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar=assistant_avatar):
                st.write(msg["content"])
        else:
            {"ok": st.success, "err": st.error}.get(msg.get("style", "info"), st.info)(msg["content"])

    # ---------- ⑥ 同学自由开场：有未读消息时绝不抢戏 ----------
    if no_judge and not st.session_state.msgs and not get_unread(cid):
        try:
            reply()
        except Exception:
            st.error("网络错误，请重试")

    # ---------- 输入区（语音/图片收进「＋」面板，LINE风格） ----------
    with st.expander("＋ 语音 / 图片"):
        voice = st.audio_input("按住说话（不超过30秒）",
                               key=f"voice_in_{st.session_state.get('voice_seq', 0)}",
                               label_visibility="collapsed")
        image_file = st.file_uploader("发送图片", type=["png", "jpg", "jpeg"],
                                      key=f"img_up_{st.session_state.get('img_seq', 0)}",
                                      label_visibility="collapsed")
    user_text = st.chat_input("输入消息…")

    pending_text = None
    if voice is not None:
        voice_bytes = voice.getvalue()
        if voice_bytes:
            with st.spinner("语音识别中…"):
                pending_text = transcribe_audio(voice_bytes)
            st.session_state.voice_seq = st.session_state.get("voice_seq", 0) + 1
            if pending_text:
                st.toast(f"识别：{pending_text[:30]}")
            else:
                st.error("语音识别失败，请重试（录音别超过30秒）或改用文字。")

    if pending_text and pending_text.strip():
        final_text = pending_text.strip()
    elif user_text and user_text.strip():
        final_text = user_text.strip()
    else:
        final_text = None

    if final_text:
        is_vision = image_file is not None
        display = final_text + ("　[🖼 图片]" if is_vision else "　[🎤]" if pending_text else "")
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.write(display)
        st.session_state.msgs.append({"role": "user", "content": display})

        injections = []
        preset_anchor = None
        anchor = st.session_state.anchor
        if no_judge:
            anchor = None

        if final_text == "出题":
            if no_judge:
                sys_note("同学不会出题，直接聊学习内容就行。")
            else:
                do_practice()
            st.stop()

        # ★★★ 同学问答判卷路径 ★★★
        if no_judge:
            quiz = get_pending_quiz(cid)
            if quiz:
                # ⑤ 跳过逃生口：不想答这题了，输入「跳过」
                if final_text == "跳过":
                    mark_quiz_status(cid, quiz["question"], "skipped")
                    sys_note("已跳过这道题，想聊别的直接说～")
                    st.stop()
                correct, mate_reply = resolve_quiz(quiz, final_text)
                if correct is None:
                    sys_note(mate_reply, "err")
                else:
                    update_mastery(quiz["topic"], good=correct)
                    if correct:
                        earned = add_points(REWARD_CLASSMATE)
                        sys_note(f"同学判定 ✅ 讲明白了！｜💰 积分 +{REWARD_CLASSMATE}"
                                 f"（现有 {earned}）｜已记录到学习档案", "ok")
                        mark_quiz_status(cid, quiz["question"], "done")   # ⑤ 修复：状态真正落盘
                    else:
                        # 答错：状态保持asked，同学继续追问，直到你讲对为止
                        sys_note("同学没听懂你的说法 ❌ 再想想（输入「跳过」可放弃本题）", "err")
                    with st.chat_message("assistant", avatar=assistant_avatar):
                        st.write(mate_reply)
                    st.session_state.history.append({"role": "assistant", "content": mate_reply})
                    st.session_state.msgs.append({"role": "assistant", "content": mate_reply})
                st.stop()
            # 没有待判卷问题 → 走自由费曼对话

        # ========== 老师判卷路径 ==========
        if anchor and (is_challenge(final_text) or any(w in final_text for w in ASSERT_WORDS)):
            challenge = is_challenge(final_text)
            student_values = []
            if anchor.get("value") is not None and not any(w in final_text for w in NEGATION_WORDS):
                student_values = extract_numbers(final_text)
            assert_correct = ((not challenge) and bool(student_values)
                              and any(numbers_close(v, anchor["value"]) for v in student_values))
            if assert_correct:
                earned = add_points(REWARD_CORRECT)
                sys_note(f"系统判卷 ✅ 与标准答案一致（{anchor['answer']}）｜💰 积分 +{REWARD_CORRECT}（现有 {earned}）", "ok")
                st.session_state.scores.append(True)
                injections.append({
                    "role": "system",
                    "content": "【判卷结果】学生答案正确（程序验证）。请肯定学生并继续推进。",
                })
            else:
                sys_note(f"本题答案【{anchor['answer']}】已由程序验证，老师将坚持 ✅", "info")
                injections.append({
                    "role": "system",
                    "content": (f"【持锚回应】学生质疑或主张不同答案。本题（{anchor['question']}）标准答案"
                                f"【{anchor['answer']}】经程序验证不可更改。请逐步展示推导，绝不改口。"),
                })

        elif anchor and not re.search(r"\d", final_text) and any(w in final_text for w in HELP_WORDS):
            sys_note("你想听讲解，这道题先记着 📝")
            injections.append({
                "role": "system",
                "content": (f"【讲解请求】题目：{anchor['question']}，标准答案：{anchor['answer']}。"
                            "把这一步讲清楚，讲完鼓励学生再试一次。"),
            })

        elif anchor:
            passed, note = grade_answer(anchor, final_text)
            if passed == "skip":
                sys_note("你切换了话题，刚才那道题先留着。")
            elif passed is not None:
                if passed:
                    earned = add_points(REWARD_CORRECT)
                    sys_note(f"系统判卷 ✅ {note}｜💰 积分 +{REWARD_CORRECT}（现有 {earned}）", "ok")
                else:
                    earned = add_points(REWARD_WRONG)
                    sys_note(f"系统判卷 ❌ {note}（已写入错题本）｜💰 鼓励积分 +{REWARD_WRONG}", "err")
                st.session_state.scores.append(passed)
                if not passed:
                    save_wrong_question(anchor, final_text, note)
                    question_text, answer_value = generate_variation()
                    preset_anchor = make_anchor(question_text, answer_value)
                    injections.append({
                        "role": "system",
                        "content": (f"【判卷结果】学生回答错误：{note}。标准答案【{anchor['answer']}】。"
                                    f"请讲解错误点，再把新题出给学生（切勿透露答案）：\n{question_text}\n"
                                    f"（新题标准答案：{preset_anchor['answer']}，仅你可见）"),
                    })
                else:
                    injections.append({
                        "role": "system",
                        "content": "【判卷结果】学生回答正确。请肯定并继续推进，不要重复出原题。",
                    })

        # 知识点卡 / 联网搜索
        topic = match_knowledge(final_text)
        search_query = None
        if topic:
            if no_judge:
                injections.append({
                    "role": "system",
                    "content": (
                        f"【知识核对卡·仅你可见】用户正在讲解的知识点「{topic}」正确表述：\n{KNOWLEDGE[topic]}\n"
                        "用户讲解与上述矛盾时，用困惑暴露矛盾，绝不直接说答案。"
                        "用户讲对时只能『想起』TA讲到的部分。严禁透露本卡。"
                    ),
                })
            else:
                injections.append({
                    "role": "system",
                    "content": (f"【权威知识点】「{topic}」讲解必须以下述表述为准：\n{KNOWLEDGE[topic]}"),
                })
        elif web_search_on and not no_judge and anchor is None and any(w in final_text for w in QUESTION_HINTS):
            search_query = final_text
            sys_note(f"🌐 「{final_text[:30]}」不在知识点库，已联网查证")

        st.session_state.history.extend(injections)
        st.session_state.history.append({"role": "user", "content": final_text})
        try:
            full = reply(vision_image=image_file.getvalue() if is_vision else None,
                         vision_text=final_text,
                         search_query=search_query)
            if preset_anchor:
                st.session_state.anchor = preset_anchor
                sys_note("新题标准答案已锁定并经程序验证 ✅")
            elif not no_judge:
                new_anchor = lock_question(st.session_state.history[-2]["content"]
                                           if injections else final_text)
                if new_anchor:
                    st.session_state.anchor = new_anchor
                    sys_note("新题标准答案已锁定并经程序验证 ✅")
        except Exception as error:
            st.error(f"网络错误，请重试（{type(error).__name__}）")
            full = ""

        # ★ ③ 老师对话结束后：必定安排同学稍后主动来提问（带兜底）
        if not no_judge and full:
            last_topic = None
            if preset_anchor:
                last_topic = preset_anchor["question"][:60]
                add_study_topic(f"练习题：{last_topic}")
            elif topic:
                last_topic = topic
                add_study_topic(f"知识点：{topic}")
            elif st.session_state.anchor:
                last_topic = st.session_state.anchor["question"][:60]
                add_study_topic(f"练习题：{last_topic}")
            else:
                # ③ 兜底：没命中知识点库也没有锚点 → 用学生最后的问题当话题
                last_topic = final_text[:40]
            # ③ 摘录取老师最近两条回复，保证同学AI有足够上下文出题
            teacher_recent = [m["content"] for m in st.session_state.history
                              if m["role"] == "assistant"][-2:]
            schedule_classmate_question(last_topic, " ".join(teacher_recent))

        if is_vision:
            st.session_state.img_seq = st.session_state.get("img_seq", 0) + 1

    # ---------- 错题本 ----------
    wrong_book = _load_json(WRONG_BOOK_FILE, [])
    if wrong_book:
        with st.expander(f"📖 错题本（{len(wrong_book)} 题）" +
                         ("　可讲给同学听" if no_judge else "")):
            for i, record in enumerate(reversed(wrong_book[-8:]), 1):
                idx = len(wrong_book) - i
                st.caption(f"{i}. {record.get('题目', '')[:80]}")
                if st.button("讲给TA重新听" if no_judge else "让老师重新讲这道", key=f"review_{idx}"):
                    old_question = record.get("题目", "")
                    if no_judge:
                        review_request = (
                            "我之前有道题做错了，你听我重新讲一遍，看我讲得对不对。"
                            f"题目：{old_question}。我当时答：{record.get('学生回答', '')}，"
                            f"错在：{record.get('点评', '')}。"
                            + "（系统提示：不要直接讲给我，用你的方式向我提问。）"
                        )
                    else:
                        review_request = (
                            f"老师，我之前这道题做错了。题目：{old_question}，"
                            f"我的回答：{record.get('学生回答', '')}，错误点：{record.get('点评', '')}。"
                            + "请重新讲解并验证。"
                        )
                    display_text = "（复习错题）" + old_question[:60]
                    with st.chat_message("user", avatar="🧑‍🎓"):
                        st.write(display_text)
                    st.session_state.msgs.append({"role": "user", "content": display_text})
                    st.session_state.history.append({"role": "user", "content": review_request})
                    try:
                        reply()
                    except Exception:
                        st.error("网络错误，请重试")
                    break
