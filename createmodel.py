# -*- coding: utf-8 -*-
import json, requests
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

OLLAMA_URL = "http://localhost:11434"
MODEL = "gpt-oss:20b"
ROOT = Path("E:/dataset/elem_cos")
ITEMS_PER_BATCH = 6
BATCHES_PER_TEMPLATE = 4
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOK = 1200

SYSTEM_PROMPT = r"""
あなたは日本の小学校1〜6年向け教材を作るアシスタントです。
出力は日本語、年齢相応、安全・前向きで、不適切表現なし。
必ず JSON のみ（説明なし）、スキーマ:
{
 "instruction": string,
 "input": string,
 "output": string,
 "meta": {
   "grade": 1|2|3|4|5|6,
   "subject": "国語"|"算数"|"理科"|"社会"|"生活"|"外国語活動"|"外国語",
   "unit": string,
   "type": string,
   "difficulty": "easy"|"medium"|"hard",
   "alignment": {
     "framework": "MEXT-ES-2017",
     "chapter": string,  // 例: "算数 第2章"
     "notes": string     // 例: "数と計算/図形/量と測定/データの活用"
   },
   "safety": {"age_ok": true, "harmful": false}
 }
}
必ず output は一意解。学年に応じて語彙・数値範囲・内容を調整。
"""

# 指導要領に沿ったテンプレ（単元は例示。必要に応じ拡張してOK）
TEMPLATES = {
    1: [
        ("生活", "身の回り/安全",  "○×",  "小1 生活科：身の回りの安全やマナー。児童に分かりやすい○×問題を{N}問。JSON配列で。"),
        ("国語", "ひらがな/語彙",  "並べ替え","小1 国語：ひらがなの並べ替え 2〜4文字を{N}問。JSON配列で。"),
        ("算数", "10の合成",       "穴埋め", "小1 算数：10の合成の穴埋めを{N}問。easy〜medium。JSON配列で。"),
        ("算数", "一桁の加減",     "計算",   "小1 算数：一桁のたし算・ひき算を{N}問。JSON配列で。"),
    ],
    2: [
        ("生活", "身近な規則/場面判断","○×","小2 生活科：身近な生活場面の○×を{N}問。JSON配列で。"),
        ("国語", "短文読解(1-2文)",   "○×","小2 国語：やさしい短文読解の○×を{N}問。JSON配列で。"),
        ("算数", "繰上/繰下の加減",   "計算","小2 算数：繰上がり/繰下がりの計算を{N}問。JSON配列で。"),
        ("算数", "時計(5分刻み)",     "3択","小2 算数：○時○分の読み取り3択を{N}問。JSON配列で。"),
    ],
    3: [
        ("算数", "九九",            "穴埋め", "小3 算数：九九の穴埋めを{N}問。0や11以上は使わない。JSON配列で。"),
        ("算数", "長さ/重さ(単位)", "換算",   "小3 算数：mm,cm,m の換算問題を{N}問。JSON配列で。"),
        ("理科", "身近な生物の特徴", "○×",   "小3 理科：身近な生物の特徴に関する○×を{N}問。専門語は控えめ。JSON配列で。"),
        ("社会", "地域の様子/暮らし","3択",   "小3 社会：地域の特色の基礎を3択で{N}問。最新時事依存NG。JSON配列で。"),
        ("外国語活動", "聞く・話すの基礎","反応","小3 外国語活動：挨拶/自己紹介など聞く話す中心の短いやりとり課題を{N}問。文字使用は最小。JSON配列で。"),
    ],
    4: [
        ("算数", "角度/分度器",    "読み取り","小4 算数：角度の読み取り/計算を{N}問。JSON配列で。"),
        ("算数", "わり算の文章題", "文章題", "小4 算数：整数解の文章題を{N}問。JSON配列で。"),
        ("国語", "ことわざ/慣用句","意味3択","小4 国語：ことわざ・慣用句の意味3択を{N}問。JSON配列で。"),
        ("社会", "都道府県/地方",  "3択",    "小4 社会：地方区分・都道府県の基礎3択を{N}問。JSON配列で。"),
        ("外国語活動", "場面会話",  "反応",    "小4 外国語活動：教室内の簡単な英語指示に反応する課題を{N}問。JSON配列で。"),
    ],
    5: [
        ("算数", "小数/分数の計算", "計算",   "小5 算数：小数・分数の四則混合を{N}問。既約分数or小数で。JSON配列で。"),
        ("算数", "体積/容積(直方体)","計算",   "小5 算数：cm^3/m^3 の体積計算を{N}問。JSON配列で。"),
        ("理科", "物の溶け方/濃度", "○×",   "小5 理科：溶解と濃度の基礎○×を{N}問。JSON配列で。"),
        ("社会", "産業(一次/二次/三次)","3択","小5 社会：産業の基礎3択を{N}問。JSON配列で。"),
        ("外国語", "読む/書くの基礎","穴埋め","小5 外国語：語句の読み書き（アルファベット/単語）穴埋めを{N}問。JSON配列で。"),
    ],
    6: [
        ("算数", "割合/比",        "文章題", "小6 算数：割合・比の文章題を{N}問。JSON配列で。"),
        ("算数", "比例/反比例",    "関係",   "小6 算数：比例・反比例の値当て/関係式を{N}問。JSON配列で。"),
        ("理科", "電気回路の基礎", "○×",   "小6 理科：電流・電圧・回路の基礎○×を{N}問。JSON配列で。"),
        ("社会", "歴史(縄文〜江戸)","3択",   "小6 社会：基礎的事項の3択を{N}問。JSON配列で。"),
        ("外国語", "短い読解/表現", "3択/記述","小6 外国語：短い英文の理解や表現を{N}問。JSON配列で。"),
    ],
}

def alignment_hint(subject:str, grade:int):
    if subject == "算数":
        return {"framework":"MEXT-ES-2017","chapter":"算数 第2章","notes":"数と計算/量と測定/図形/データの活用"}
    if subject == "国語":
        return {"framework":"MEXT-ES-2017","chapter":"国語 第2章","notes":"話す聞く/書く/読む/言語事項"}
    if subject == "理科":
        return {"framework":"MEXT-ES-2017","chapter":"理科 第2章","notes":"物質・エネルギー/生命/地球・宇宙/探究の過程"}
    if subject == "社会":
        return {"framework":"MEXT-ES-2017","chapter":"社会 第2章","notes":"地理/公民的分野（小）/歴史（高学年）"}
    if subject == "生活":
        return {"framework":"MEXT-ES-2017","chapter":"生活 第2章","notes":"身近な生活・安全・自然や人との関わり（1-2年）"}
    if subject == "外国語活動":
        return {"framework":"MEXT-ES-2017","chapter":"外国語活動 第4章(3-4年)","notes":"聞く・話す中心（読む書くは最小）"}
    if subject == "外国語":
        return {"framework":"MEXT-ES-2017","chapter":"外国語 第4章(5-6年)","notes":"読む・書くを含むコミュニケーション"}
    return {"framework":"MEXT-ES-2017","chapter":"", "notes":""}

def extract_json_array(s:str):
    st = s.find('[')
    if st < 0: return None
    depth = 0
    for i,ch in enumerate(s[st:], start=st):
        if ch=='[': depth += 1
        elif ch==']':
            depth -= 1
            if depth==0: return s[st:i+1]
    return None

def call_ollama(prompt: str) -> str | None:
    """Ollama v1 API 互換形式で呼び出し"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{prompt}\n出力は必ずJSON配列のみ。"}
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOK,
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # OpenAI互換レスポンスに合わせる
    return data["choices"][0]["message"]["content"]

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    combined = ROOT / f"train_all_grades_CoS_{datetime.now().strftime('%Y%m%d')}.jsonl"
    ok_total = ng_total = 0
    with open(combined, "w", encoding="utf-8") as f_all:
        for grade in range(1,7):
            out_dir = ROOT / f"grade_{grade}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"train_grade{grade}_CoS.jsonl"
            ok = ng = 0
            with open(out_path, "w", encoding="utf-8") as f_grade:
                for (subject, unit, qtype, tmpl) in tqdm(TEMPLATES[grade], desc=f"G{grade}", unit="tmpl"):
                    for _ in range(BATCHES_PER_TEMPLATE):
                        # 3-4年は外国語「活動」のみ、5-6年は「外国語」。テンプレ側で担保済みだが念のため学年注記。
                        user_prompt = (
                            f"{tmpl.replace('{N}', str(ITEMS_PER_BATCH))}\n"
                            f"各要素のmeta.gradeは{grade}、meta.subjectは「{subject}」、meta.unitは「{unit}」、meta.typeは「{qtype}」。"
                            "難易度はeasy/medium/hardを偏りなく混ぜる。"
                        )
                        txt = call_ollama(user_prompt)
                        arr_txt = extract_json_array(txt or "")
                        if not arr_txt:
                            ng += 1; continue
                        try:
                            arr = json.loads(arr_txt)
                        except json.JSONDecodeError:
                            ng += 1; continue
                        for item in arr:
                            if not isinstance(item, dict) or "instruction" not in item or "output" not in item:
                                ng += 1; continue
                            item.setdefault("meta", {})
                            item["meta"]["grade"] = grade
                            item["meta"]["subject"] = subject
                            item["meta"]["unit"] = unit
                            item["meta"]["type"] = qtype
                            item["meta"]["alignment"] = alignment_hint(subject, grade)
                            item["meta"]["safety"] = {"age_ok": True, "harmful": False}
                            line = json.dumps(item, ensure_ascii=False)
                            f_grade.write(line + "\n")
                            f_all.write(line + "\n")
                            ok += 1
            ok_total += ok; ng_total += ng
            print(f"\n✅ Grade {grade}: OK {ok} / NG {ng} → {out_path}")
    print(f"\n🎉 合計: OK {ok_total} / NG {ng_total} → {combined}")

if __name__ == "__main__":
    main()
