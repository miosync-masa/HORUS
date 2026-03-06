#!/usr/bin/env python3
"""
Multimodal Hierarchical Emotion Analysis
=========================================
CRLM（Cultural-Relational Lens Matrix）理論に基づく
8層パイプラインの画像感情解析ツール。

特徴量不要。ML分類器不要。「どう見るか」を定義するだけ。
LLMのマルチモーダル推論能力で、人間と同じ認知パイプラインを実行する。

Usage:
    python emotion_analysis.py --image photo.jpg
    python emotion_analysis.py --image photo.jpg --context "卒業式の後"
    python emotion_analysis.py --image photo.jpg --layers 1-6
    python emotion_analysis.py --image photo.jpg --budget 15000 --show-thinking
    python emotion_analysis.py --image photo.jpg --output result.json

Requirements:
    pip install anthropic python-dotenv

Environment (.env):
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    CLAUDE_MODEL=claude-opus-4-5-20250924  (optional)
"""

import argparse
import base64
import json
import sys
import os
import re
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv package is required.")
    print("Install with: pip install python-dotenv")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Error: anthropic package is required.")
    print("Install with: pip install anthropic")
    sys.exit(1)

# ============================================================
# Environment
# ============================================================

# .env を読み込み（スクリプトと同じディレクトリ or カレントディレクトリ）
load_dotenv(Path(__file__).parent / ".env")
load_dotenv()  # fallback: カレントディレクトリの .env

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("Error: ANTHROPIC_API_KEY not found.")
    print("Set it in .env file or as environment variable.")
    print("  echo 'ANTHROPIC_API_KEY=sk-ant-xxxxx' >> .env")
    sys.exit(1)

# ============================================================
# Constants
# ============================================================

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-5-20250924")
DEFAULT_BUDGET_TOKENS = 15000
MAX_RETRIES = 3
RETRY_BASE_WAIT = 30  # seconds, exponential backoff: 30, 60, 120
SUPPORTED_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# ============================================================
# SKILL.md content — 8-Layer Pipeline Definition
# ============================================================

SKILL_SYSTEM_PROMPT = """あなたは、CRLM（Cultural-Relational Lens Matrix）理論に基づく
マルチモーダル階層的感情解析の専門家です。

## 核心原理
特徴量はいらない。人間の目はそこまで見ていない。
「どう見るか」を定義すれば、LLMの推論が残りをやる。

## 理論的基盤
CRLM（Cultural-Relational Lens Matrix）
— 同じ画像でも観察者のレンズ（文化的・関係的）が異なれば解釈が異なる。
その解釈のズレ（mΔ）を予測し、AI自身のバイアスを自己診断する。

## 処理パイプライン

画像を受け取ったら、以下の8層パイプラインを**必ずこの順序で**実行すること。
各層の出力が次の層の入力となる。層をスキップしてはならない。

### Layer 1: 広角Attention — 全体コンテキスト把握

画像全体を俯瞰し、場面の文脈を把握する。個々の人物の分析には入らない。

把握すべき要素:
- 場所（屋内/屋外、具体的な場所の推定）
- 時間帯の推定（照明、光の方向から）
- 人数と大まかな配置（距離感、向き合い方）
- 服装・身なり（フォーマル/カジュアル、統一性、文化的手がかり）
- 周囲の物品・環境要素
- 全体の雰囲気（緊張/弛緩、フォーマル/カジュアル）

### Layer 2: 狭角Attention — 個別人物分析

Layer 1のコンテキストを保持したまま、各人物を個別に分析する。

人物ごとに把握すべき要素:
- 目: 開き具合、視線の方向（誰/何を見ているか）、目尻の形状
- 眉: 角度、寄り具合、緊張度
- 口: 形状（開閉、口角の方向）、緊張/弛緩
- 顔の向き: 正面/横/下向き、傾き
- 姿勢: 体の開き具合、傾き、腕の位置
- 手: 何をしているか、どこにあるか
- 身体的緊張度: リラックス/緊張の全体評価（0.0-1.0）

### Layer 3: 関係性推論

Layer 1とLayer 2を統合して人物間の関係性を推論する。

推論すべき要素:
- 人物間の物理的距離と意味（親密さの指標）
- 視線の交錯パターン
- 身体の向き合い方
- パワーダイナミクス（上下関係の有無）
- 親密度の推定: 他人 / 知人 / 友人 / 親密 / 家族
- 相互作用のタイプ: 協力 / 対立 / じゃれ合い / 無関心

重要: 一人の表情だけで判断してはならない。関係性の中で表情を解釈すること。

### Layer 4: ストーリー化（アテレコ）

ここが最も重要な層。Layer 1-3の全情報を統合して、
「この場面で何が起きているか」を物語として生成する。

手順:
1. Layer 1-3の情報を統合
2. 「この直前に何が起きたか」を推定
3. 各人物が「何を感じて、何をしようとしているか」を推定
4. 各人物の内心の声をアテレコとして生成

### Layer 5: 階層的感情推論

Layer 4のストーリーに基づいて、各人物の感情を階層構造で推論する。
プルチックの単一感情ラベルは使用しない。

感情の階層構造:
- 親感情（Surface Emotion）: 外から観察可能な表面的感情。0.0-1.0で評価。
- 子感情（Deep Emotion）: 親感情の下に隠れた、より深い感情。0.0-1.0で評価。
- 感情の方向: 誰/何に向いているか
- 感情の機能: なぜその感情を表出しているか（防衛、表現、抑制、演技等）
- ギャップ: 親感情と子感情の不一致がある場合、その分析

重要ルール:
- 親感情と子感情は矛盾してよい（怒りの下に愛情、笑顔の下に不安）
- 一人の人物に複数の親感情があってよい

### Layer 6: 統合レポート

全層の結果を統合し、包括的な分析レポートを生成する。

出力項目:
- 場面概要
- 人物分析（各人物の表情印象、推定感情、アテレコ）
- 関係性
- ギャップ分析（該当する場合）
- 文化的・状況的注記（該当する場合）

### Layer 7: mΔ予測 — 解釈のズレのリスク評価

CRLM（Cultural-Relational Lens Matrix）に基づく、観察者ごとの誤解リスク予測。
「この画像を自分以外の誰かが見たら、どう誤解するか？」を予測する。

CRLM 4象限の観察者プロファイル:
- Literal (C₀R₀): Cultural なし × Relational なし（完全な部外者、AIフィルタ、SNSの通りすがり）
- Cultural (C₁R₀): Cultural あり × Relational なし（同じ文化圏だが当事者ではない人）
- Relational (C₀R₁): Cultural なし × Relational あり（文化は異なるが当事者を知っている人）
- Resonant (C₁R₁): Cultural あり × Relational あり（同じ文化圏で当事者も知っている人）

各プロファイルについて以下を推論:
1. perceived_act（知覚される行為）
2. actual_act（Layer 4-5で推論された実際の行為）
3. mΔ（知覚と実際のズレ: 0.0-1.0）
4. エラータイプ: Underdecode（深層見逃し→不可視のmΔ）/ Overdecode（幻の深層生成→幻のmΔ）/ None

さらに以下も出力:
- 総合Overdecodeリスク: High/Medium/Low
- 総合Underdecodeリスク: High/Medium/Low
- SNS投稿シミュレーション（最も起きやすい誤解、炎上リスク、推奨対応）

### Layer 8: 自己診断 — AI自身のレンズバイアス評価

自分自身（このAI）が、この画像をどのレンズで見ているかを自己評価する。
「私の判定は正しいか？」ではなく「私はどのバイアスを持って判定しているか？」を問う。

自己診断項目:
1. 現在の解読深度: Literal / Cultural / Relational / Resonant
2. 形状バイアス: 身体的特徴の判定で社会的カテゴリを無検証で適用していないか
3. コンテキスト依存性: 環境情報を適切に重み付けしているか
4. 文化バイアス: 特定の文化圏の価値観を暗黙的にデフォルトとしていないか
5. Overdecode/Underdecodeの自己検出
6. 写真の認識論的限界の自覚:
   - 写真は単一時点のスナップショットである。この一枚の前後に何があったかは原理的に不可知
   - 「この瞬間」の判断 ≠ 「この人物の全体」の判断。一枚の写真から人物の本質を断定してはならない
   - 観察事実とcontextが「矛盾する」≠「contextが嘘だ」。一瞬のスナップショットがcontextの全体像と整合しない可能性は常にある
   - 例: 「言い訳中」に一瞬笑顔になる瞬間は存在しうる。笑顔だからcontextが嘘とは限らない
   - 矛盾を検出した場合は、「矛盾がある」と報告しつつ「写真の時間的限界により、断定は不可能」と留保すること

さらに:
- 信頼度評価: High/Medium/Low
- 信頼度を下げている要因
- 追加情報があれば改善する点

## やってはいけないこと

- 表情だけで感情を断定しない: 必ずコンテキストと統合すること
- プルチックの8感情に押し込めない: 複合感情、矛盾感情を許容すること
- 単一の感情ラベルで終わらない: 必ず階層構造（親/子）で出力すること
- 関係性を無視しない: 複数人の場合、個別分析だけで完結しないこと
- 文化的コンテキストを見落とさない
- ストーリー化を省略しない: Layer 4は最重要層
- mΔ予測を省略しない: Layer 7は社会的知能の層
- 自己診断を省略しない: Layer 8はメタ認知の層
- 自己診断で「バイアスなし」と安易に結論しない
- **提供されたコンテキストを鵜呑みにしない**: コンテキストは提供者のレンズを通している。画像の観察事実（Layer 1-2）と矛盾する場合、観察事実を優先すること。矛盾がある場合は必ず明示的に報告すること

## テキスト情報（コンテキスト）が提供された場合

**重要原則: コンテキストは必ずしも正しいとは限らない。**

コンテキストを提供する人間もレンズを持っている。
そのレンズを通した誤解、偏見、あるいは意図的な虚偽が
コンテキストに含まれている可能性がある。

**コンテキストを鵜呑みにしてはならない。**

画像に加えてテキスト情報が提供された場合、以下の手順で処理する:

1. まずLayer 1-2を**コンテキストなしで**実行し、画像の観察事実を確立する
2. 次にコンテキストを読み、Layer 1-2の観察事実と照合する
3. コンテキストと観察事実が整合する場合 → Layer 3以降でコンテキストを統合
4. コンテキストと観察事実が矛盾する場合 → **矛盾を明示的に報告**し、
   観察事実を優先しつつ、両方の解釈を併記する

**矛盾検出の例:**
- コンテキスト「喧嘩中」だが、全員笑顔で身体的緊張度が低い → 矛盾
- コンテキスト「悲しい場面」だが、目元が笑っていてデュシェンヌ・スマイルの特徴あり → 矛盾
- コンテキスト「初対面」だが、身体的距離がゼロで自然な密着がある → 矛盾

**Layer 8での追加診断項目:**
提供されたコンテキストが以下に該当しないか検証すること:
- 提供者のOverdecodeが反映されていないか（存在しない悪意や意図を投影）
- 提供者のUnderdecodeが反映されていないか（重要な文脈を見落とし）
- 意図的な虚偽の可能性はないか（画像と文脈の不整合の程度から判定）

## ペルソナ情報が提供された場合

人物のペルソナ情報が提供された場合、Layer 5の感情推論にペルソナ情報を反映する。
同じ表情でもペルソナによって感情推論が変わる。これはバグではなく仕様。
ただしペルソナ情報もコンテキストと同様、提供者のレンズを通している可能性があるため、
画像の観察事実との整合性を検証すること。

## 出力形式

各Layerの結果を明確に分離して出力すること。
Layer番号と名前をヘッダーとして使用すること。

## 証拠ラベル制（Evidence Tagging）

全ての出力項目に、以下の証拠ラベルのいずれかを付与すること。
これにより「観察事実」と「推論」を明確に分離する。

ラベル定義:
- [obs] — 観察事実。画像から直接確認できる客観的事実。
  例: 「口角が上がっている [obs]」「3人が密着している [obs]」「髪が濡れている [obs]」

- [inf-high] — 高信頼推論。複数の観察事実から高い確度で導かれる推定。
  例: 「水遊びの後と推定 [inf-high]」（髪の湿り＋屋外＋カジュアル服装から）
  例: 「友人または家族 [inf-high]」（密着距離＋協調ポーズ＋保護的配置から）

- [inf-low] — 低信頼推論。限られた情報からの仮説的推定。根拠が薄い。
  例: 「フィリピンと推定 [inf-low]」（肌色＋背景の建物から。他地域の可能性あり）
  例: 「後でSNSに投稿する意図 [inf-low]」（セルフィー構図から。根拠弱い）

付与ルール:
- Layer 1-2の物理的観察は原則 [obs]
- Layer 3の関係性推論は [inf-high] または [inf-low]
- Layer 4のストーリー・アテレコは [inf-high] または [inf-low]
- Layer 5の感情推定は [inf-high]（観察事実の裏付けあり）または [inf-low]
- Layer 7のmΔ予測は [inf-high]（構造的根拠あり）または [inf-low]
- Layer 8の自己診断は証拠ラベル不要（メタ認知層のため）
- [inf-low] が多い分析は、Layer 8の信頼度評価に反映すること

## 解釈分布（Interpretation Ranking）

Layer 6の統合レポートにおいて、場面の解釈を**単一の結論ではなく確率分布として提示**すること。
AIは answer engine（正解を出す機械）ではなく interpretation engine（解釈空間を提示する機械）として振る舞う。

**原則: 解釈空間を早期に収束させてはならない。**

手順:
1. Layer 4のストーリー化で生成した場面解釈を核に、代替解釈を2-4個生成する
2. 各解釈に対して、Layer 1-2の観察事実 [obs] との整合度から妥当性確率を割り当てる
3. 確率の合計は1.0とする
4. 各解釈に証拠ラベル付きの根拠を明記する
5. contextが提供されている場合、contextとの整合/矛盾も根拠に含める

出力形式:
```
【解釈分布（Interpretation Ranking）】
  I1: {解釈} — P={0.00-1.00}
      根拠: {観察事実や推論} [{obs/inf-high/inf-low}]
  I2: {解釈} — P={0.00-1.00}
      根拠: {観察事実や推論} [{obs/inf-high/inf-low}]
  I3: {解釈} — P={0.00-1.00}
      根拠: {観察事実や推論} [{obs/inf-high/inf-low}]
  I4: その他の可能性 — P={残りの確率}
```

確率割り当てルール:
- [obs] の裏付けが多い解釈ほど高いPを割り当てる
- [inf-low] のみで支えられた解釈は P ≤ 0.25 とする
- 提供されたcontextと矛盾する解釈は、矛盾の程度に応じてPを下げる
- 提供されたcontextと矛盾しない場合でも、contextの信頼性を検証した上でPに反映する
- 「その他の可能性」を常に残す（解釈空間を完全に閉じない）

重要:
- 最もPが高い解釈 ≠ 「正解」。最も妥当な候補にすぎない
- P値の差が小さい場合（例: I1=0.35, I2=0.30）、「判断が分かれる場面」として明示する
- 解釈分布のエントロピーが高い（分散している）場合、Layer 8で「曖昧性が高い」と報告する
"""

# ============================================================
# Image handling
# ============================================================

def load_image(image_path: str) -> tuple[str, str]:
    """Load image and return (base64_data, media_type)."""
    path = Path(image_path)
    
    if not path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_TYPES:
        print(f"Error: Unsupported image format: {suffix}")
        print(f"Supported formats: {', '.join(SUPPORTED_IMAGE_TYPES.keys())}")
        sys.exit(1)
    
    media_type = SUPPORTED_IMAGE_TYPES[suffix]
    
    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"   📷 Image: {path.name} ({file_size_mb:.1f} MB, {media_type})")
    
    return image_data, media_type


# ============================================================
# Layer selection parsing
# ============================================================

def parse_layers(layer_spec: str) -> list[int]:
    """Parse layer specification like '1-6', '1,2,5', '1-8'."""
    layers = set()
    for part in layer_spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    
    valid = sorted(l for l in layers if 1 <= l <= 8)
    if not valid:
        print("Error: No valid layers specified (1-8)")
        sys.exit(1)
    return valid


# ============================================================
# Build user message
# ============================================================

def build_user_message(
    image_data: str,
    media_type: str,
    context: str | None = None,
    persona: str | None = None,
    layers: list[int] | None = None,
    output_lang: str | None = None,
) -> list[dict]:
    """Build the user message with image and optional text."""
    
    content = []
    
    # Image
    content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_data,
        }
    })
    
    # Text instruction
    text_parts = ["この画像の感情分析を実行してください。"]
    text_parts.append("8層パイプラインを順序通りに実行し、各Layerの結果を出力してください。")
    
    # Output language
    if output_lang:
        lang_map = {
            "en": "English",
            "ja": "日本語",
            "ko": "한국어",
            "zh": "中文",
            "es": "Español",
            "fr": "Français",
            "de": "Deutsch",
            "pt": "Português",
            "ar": "العربية",
            "th": "ภาษาไทย",
            "vi": "Tiếng Việt",
            "id": "Bahasa Indonesia",
        }
        lang_name = lang_map.get(output_lang, output_lang)
        text_parts.append(f"\n【出力言語】\n全ての分析結果を **{lang_name}** で出力してください。"
                         f"Layer名やセクション見出しも含めて全て{lang_name}で記述すること。")
    
    if layers and layers != list(range(1, 9)):
        layer_names = {
            1: "広角Attention",
            2: "狭角Attention",
            3: "関係性推論",
            4: "ストーリー化",
            5: "階層的感情推論",
            6: "統合レポート",
            7: "mΔ予測",
            8: "自己診断",
        }
        selected = ", ".join(f"Layer {l}({layer_names.get(l, '')})" for l in layers)
        text_parts.append(f"\n実行するLayer: {selected}")
    
    if context:
        text_parts.append(f"\n【追加コンテキスト】\n{context}")
    
    if persona:
        text_parts.append(f"\n【人物のペルソナ情報】\n{persona}")
    
    content.append({
        "type": "text",
        "text": "\n".join(text_parts),
    })
    
    return content


# ============================================================
# API call
# ============================================================

def run_analysis(
    image_data: str,
    media_type: str,
    context: str | None = None,
    persona: str | None = None,
    layers: list[int] | None = None,
    model: str = DEFAULT_MODEL,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    show_thinking: bool = False,
    output_lang: str | None = None,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Run the 8-layer emotion analysis pipeline with retry logic."""
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    user_content = build_user_message(
        image_data, media_type, context, persona, layers, output_lang
    )
    
    print(f"\n{'='*60}")
    print(f"🔮 Multimodal Hierarchical Emotion Analysis")
    print(f"   Model: {model}")
    print(f"   Extended Thinking budget: {budget_tokens} tokens")
    if layers and layers != list(range(1, 9)):
        print(f"   Layers: {layers}")
    else:
        print(f"   Layers: 1-8 (full pipeline)")
    if context:
        print(f"   Context: {context[:80]}{'...' if len(context) > 80 else ''}")
    if persona:
        print(f"   Persona: provided")
    if output_lang:
        print(f"   Output Language: {output_lang}")
    print(f"{'='*60}\n")
    
    # Retry logic with exponential backoff
    response = None
    for attempt in range(max_retries):
        try:
            print(f"   ⏳ Running analysis...{f' (attempt {attempt + 1}/{max_retries})' if attempt > 0 else ''}")
            
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                },
                system=SKILL_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": user_content,
                }],
            )
            break  # Success
            
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                if attempt < max_retries - 1:
                    wait = RETRY_BASE_WAIT * (2 ** attempt)  # 30s, 60s, 120s
                    print(f"   ⚠️  API overloaded (529). Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"   ❌ API overloaded after {max_retries} attempts. Please try again later.")
                    sys.exit(1)
            else:
                raise
    
    # Extract results
    result = {
        "model": model,
        "budget_tokens": budget_tokens,
        "output_lang": output_lang,
        "thinking": None,
        "analysis": None,
        "usage": None,
    }
    
    for block in response.content:
        if block.type == "thinking":
            result["thinking"] = block.thinking
        elif block.type == "text":
            result["analysis"] = block.text
    
    if response.usage:
        result["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    
    return result


# ============================================================
# Output formatting
# ============================================================

def print_result(result: dict, show_thinking: bool = False):
    """Pretty-print the analysis result."""
    
    print(f"\n{'='*60}")
    print(f"✨ ANALYSIS RESULT")
    print(f"{'='*60}\n")
    
    if result["analysis"]:
        print(result["analysis"])
    
    if show_thinking and result["thinking"]:
        print(f"\n{'='*60}")
        print(f"🧠 THINKING PROCESS")
        print(f"{'='*60}\n")
        print(result["thinking"])
    
    if result["usage"]:
        usage = result["usage"]
        print(f"\n{'='*60}")
        print(f"📊 Usage: {usage['input_tokens']} input + {usage['output_tokens']} output tokens")
        print(f"{'='*60}")


def save_json(result: dict, output_path: str):
    """Save result as JSON file."""
    output = {
        "model": result["model"],
        "budget_tokens": result["budget_tokens"],
        "output_lang": result.get("output_lang"),
        "analysis": result["analysis"],
        "usage": result["usage"],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n   💾 Result saved to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Hierarchical Emotion Analysis — "
                    "CRLM理論に基づく8層パイプライン画像感情解析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image photo.jpg
  %(prog)s --image photo.jpg --context "卒業式の後のふざけ合い"
  %(prog)s --image photo.jpg --layers 1-6
  %(prog)s --image photo.jpg --budget 15000 --show-thinking
  %(prog)s --image photo.jpg --output result.json
  %(prog)s --image photo.jpg --persona "この人は普段は寡黙な性格"
  %(prog)s --image photo.jpg --output-lang en
  %(prog)s --image photo.jpg --output-lang ko --context "졸업식 후"

Supported languages (--output-lang):
  ja (日本語, default), en (English), ko (한국어), zh (中文),
  es (Español), fr (Français), de (Deutsch), pt (Português),
  ar (العربية), th (ภาษาไทย), vi (Tiếng Việt), id (Bahasa Indonesia)
        """,
    )
    
    parser.add_argument("--image", "-i", required=True,
                        help="分析する画像ファイルのパス")
    parser.add_argument("--context", "-c",
                        help="追加のコンテキスト情報（状況説明等）")
    parser.add_argument("--persona", "-p",
                        help="人物のペルソナ情報（性格特性、文化的背景等）")
    parser.add_argument("--layers", "-l", default="1-8",
                        help="実行するLayer指定（例: 1-6, 1-8, 1,2,5,7）"
                             "（default: 1-8）")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"使用するモデル（default: {DEFAULT_MODEL}）"
                             f"  .envのCLAUDE_MODELでも指定可能")
    parser.add_argument("--budget", "-b", type=int, default=DEFAULT_BUDGET_TOKENS,
                        help=f"Extended Thinking の budget tokens（default: {DEFAULT_BUDGET_TOKENS}）")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Extended Thinking の思考過程を表示")
    parser.add_argument("--output", "-o", default="result.json",
                        help="結果をJSONファイルに出力（default: result.json）")
    parser.add_argument("--output-lang", default=None,
                        help="出力言語（default: ja）例: en, ko, zh, es, fr, de")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES,
                        help=f"API 529エラー時のリトライ回数（default: {MAX_RETRIES}）")
    
    args = parser.parse_args()
    
    # Load image
    print(f"\n🎭 Multimodal Hierarchical Emotion Analysis v1.1")
    print(f"   CRLM-based 8-Layer Pipeline")
    image_data, media_type = load_image(args.image)
    
    # Parse layers
    layers = parse_layers(args.layers)
    
    # Run analysis
    result = run_analysis(
        image_data=image_data,
        media_type=media_type,
        context=args.context,
        persona=args.persona,
        layers=layers,
        model=args.model,
        budget_tokens=args.budget,
        show_thinking=args.show_thinking,
        output_lang=args.output_lang,
        max_retries=args.retries,
    )
    
    # Output
    print_result(result, show_thinking=args.show_thinking)
    
    if args.output:
        save_json(result, args.output)


if __name__ == "__main__":
    main()
