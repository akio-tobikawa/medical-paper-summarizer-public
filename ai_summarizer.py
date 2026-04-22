"""
AI要約モジュール（Claude API版）

Anthropic Claude APIを使用して、各論文の批判的要約を
日本語で生成する。モデルフォールバックチェーン対応。
感染症科・AMS専用プロンプト実装済み。
"""

import time
import logging
from typing import Optional

import anthropic

from pubmed_searcher import Paper

logger = logging.getLogger(__name__)


class AISummarizer:
    """AI論文要約クラス（Claude API）"""

    def __init__(self, config: dict, api_key: str):
        """
        初期化

        Args:
            config: config.yamlから読み込んだ設定辞書
            api_key: Anthropic APIキー（ANTHROPIC_API_KEY）
        """
        self.config = config
        self.specialty_name = config.get("specialty_name", "感染症科・AMS")
        self.ai_config = config.get("ai", {})
        self.model_chain = self.ai_config.get("model_chain", [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001"
        ])
        self.max_retries = self.ai_config.get("max_retries", 3)
        self.retry_delay = self.ai_config.get("retry_delay", 5)
        self.timeout = self.ai_config.get("timeout", 120)

        self.client = anthropic.Anthropic(api_key=api_key)

    def _call_with_fallback(self, prompt: str) -> Optional[str]:
        """
        フォールバックチェーンでAPIを呼び出す

        上位モデルからエラー/レート制限時に下位モデルへ順にフォールバック

        Args:
            prompt: 送信するプロンプト

        Returns:
            生成されたテキスト（全モデル失敗時はNone）
        """
        for model_name in self.model_chain:
            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"モデル {model_name} を使用中"
                        f"（試行 {attempt + 1}/{self.max_retries}）"
                    )
                    message = self.client.messages.create(
                        model=model_name,
                        max_tokens=4096,
                        temperature=0.3,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )

                    if message.content and len(message.content) > 0:
                        text = message.content[0].text
                        if text:
                            logger.info(f"[OK] {model_name} で生成成功")
                            return text

                except anthropic.RateLimitError as e:
                    logger.warning(
                        f"モデル {model_name} でレート制限"
                        f"（試行 {attempt + 1}）: {e}"
                    )
                    logger.info("→ 次のモデルに早めにフォールバックします")
                    break

                except anthropic.APIStatusError as e:
                    error_msg = str(e).lower()
                    logger.warning(
                        f"モデル {model_name} でAPIエラー"
                        f"（試行 {attempt + 1}）: {e}"
                    )
                    # モデル未対応・404はすぐ次へ
                    if e.status_code in (404, 400):
                        logger.info("→ 次のモデルに早めにフォールバックします")
                        break
                    # 504/タイムアウトは次へ
                    if e.status_code in (504, 529):
                        logger.info("→ 過負荷のため次のモデルへフォールバックします")
                        break
                    # その他はリトライ
                    if attempt < self.max_retries - 1:
                        wait = self.retry_delay * (attempt + 1)
                        logger.info(f"  {wait}秒後にリトライ...")
                        time.sleep(wait)

                except anthropic.APIConnectionError as e:
                    logger.warning(
                        f"モデル {model_name} で接続エラー"
                        f"（試行 {attempt + 1}）: {e}"
                    )
                    if attempt < self.max_retries - 1:
                        wait = self.retry_delay * (attempt + 1)
                        logger.info(f"  {wait}秒後にリトライ...")
                        time.sleep(wait)

                except Exception as e:
                    logger.warning(
                        f"モデル {model_name} で予期しないエラー"
                        f"（試行 {attempt + 1}）: {e}"
                    )
                    if attempt < self.max_retries - 1:
                        wait = self.retry_delay * (attempt + 1)
                        logger.info(f"  {wait}秒後にリトライ...")
                        time.sleep(wait)

        logger.error("全モデルで生成に失敗しました")
        return None

    def summarize_papers(
        self, papers: list[Paper], detailed_top_n: int = 10
    ) -> list[Paper]:
        """
        論文リストの要約を生成する

        Args:
            papers: 優先度順にソートされた論文リスト
            detailed_top_n: 詳細要約を行う上位論文数

        Returns:
            要約が付与された論文リスト
        """
        if detailed_top_n is None:
            detailed_top_n = self.config.get("search", {}).get(
                "detailed_top_n", 3
            )

        total = len(papers)
        logger.info(
            f"{total}件の論文を要約します"
            f"（詳細: {min(detailed_top_n, total)}件）"
        )

        for i, paper in enumerate(papers):
            is_detailed = (i < detailed_top_n)
            mode_str = "詳細" if is_detailed else "簡潔"

            logger.info(
                f"[{i+1}/{total}] {mode_str}要約中: "
                f"{paper.title[:50]}..."
            )

            prompt = self._build_prompt(paper, is_detailed)
            result = self._call_with_fallback(prompt)

            if result:
                paper.summary = {
                    "mode": "detailed" if is_detailed else "brief",
                    "content": result
                }
            else:
                paper.summary = {
                    "mode": "detailed" if is_detailed else "brief",
                    "content": "⚠ 要約の生成に失敗しました。"
                }

            # API呼び出し間隔
            if i < total - 1:
                time.sleep(2)

        return papers

    def _build_paper_info(self, paper: Paper) -> str:
        """論文基本情報ブロックを構築する"""
        if len(paper.authors) > 5:
            author_str = ", ".join(paper.authors[:5]) + " et al."
        else:
            author_str = ", ".join(paper.authors)
        pub_type_str = ", ".join(paper.pub_types) if paper.pub_types else "不明"
        return f"""【論文情報】
タイトル: {paper.title}
著者: {author_str}
ジャーナル: {paper.journal}
出版日: {paper.pub_date}
論文タイプ: {pub_type_str}
DOI: {paper.doi if paper.doi else "N/A"}
MeSH用語: {", ".join(paper.mesh_terms[:10]) if paper.mesh_terms else "N/A"}

【アブストラクト】
{paper.abstract}""".strip()

    def _detect_paper_type(self, paper: Paper) -> str:
        """
        論文タイプを判定する

        Returns:
            "guideline" / "synthesis" / "review" / "research"
        """
        types = set(paper.pub_types)
        if types & {"Practice Guideline", "Guideline"}:
            return "guideline"
        if types & {"Systematic Review", "Meta-Analysis"}:
            return "synthesis"
        if types & {"Review"}:
            return "review"
        # pub_typesが空またはキーワードでフォールバック
        text = (paper.title + " " + paper.abstract).lower()
        if "guideline" in text or "recommendation" in text:
            return "guideline"
        if "systematic review" in text or "meta-analysis" in text:
            return "synthesis"
        if "review" in text and "randomized" not in text:
            return "review"
        return "research"

    def _build_prompt(self, paper: Paper, detailed: bool) -> str:
        """論文タイプに応じてプロンプトを振り分ける"""
        paper_type = self._detect_paper_type(paper)
        logger.info(f"論文タイプ判定: {paper_type} （{paper.title[:40]}...）")
        if paper_type == "guideline":
            return self._build_guideline_prompt(paper)
        elif paper_type == "synthesis":
            return self._build_synthesis_prompt(paper)
        elif paper_type == "review":
            return self._build_review_prompt(paper)
        else:
            paper_info = self._build_paper_info(paper)
            if detailed:
                return self._build_detailed_prompt(paper_info)
            else:
                return self._build_brief_prompt(paper_info)

    def _build_detailed_prompt(self, paper_info: str) -> str:
        """詳細要約プロンプト（AMS/感染症専門医向け）"""
        return f"""あなたは感染症専門医・抗菌薬適正使用支援（AMS）薬剤師として最高水準の批判的論文評価能力を持ちます。
以下の論文について、忙しい{self.specialty_name}の専門家が短時間で本質をつかめる形式で、日本語で詳細な批判的要約を作成してください。

{paper_info}

以下の形式で出力してください。各セクションは明確に分けてください。
※「承知いたしました」「要約します」等の前置き・挨拶は一切含めず、いきなり「## サマリーインデックス情報」から出力してください。

## サマリーインデックス情報
- **重要度**: [重要度の基準]に従い、★を5つ並べて表記（例：★★★★★）
- **結論**: [40文字以内]で、この論文が何を示したか
- **実用**: [50文字以内]で、明日のAMS/感染症診療にどう活きるか。菌種名、抗菌薬名、具体的な数値などの重要キーワードは必ず**太字**にすること

## まず一言で
この論文が何を示したのかを1〜2文で日本語要約してください。

## 研究の概要
- **研究背景**: なぜこの研究が行われたか（どの臨床課題を解決しようとしたか）
- **研究デザイン**: どのような研究手法か（RCT、前向きコホート、後ろ向き多施設研究等）
- **対象患者**: どのような患者が対象か（人数・感染症の種類・重症度・施設背景）
- **介入/比較**: 何を比較したか（抗菌薬レジメン、投与期間、AMS介入の有無等）
- **主要評価項目**: 何を評価したか（臨床的治癒、30日死亡率、再発率等）
- **主な結果**: 主要な数値結果（ハザード比、95%CI、NNT、p値等を含む）

## AMS・感染症臨床的ポイント
- 抗菌薬適正使用の観点で何が重要か
- どの菌種・どの感染症でどのレジメンが支持されるか
- de-escalation・IV→PO切り替え・投与期間短縮の根拠になるか
- 耐性菌対策・カルバペネムスペアリングへの示唆
- 院内感染対策（IPC）への影響はあるか
- PK/PD的な考察（AUC/MIC、T>MIC等）が重要な場合は記載

## 限界
- バイアスの可能性（選択バイアス、交絡因子等）
- 一般化可能性の限界（市中病院 vs 大学病院、耐性菌状況の地域差）
- 日本の感染症状況・抗菌薬耐性パターンへの適用可能性
- サンプルサイズ・検出力の問題
- 観察研究の場合は因果推定の限界

## 日本の実臨床・AMS活動への実践メモ
- 明日からのAMSラウンド・カルテレビューで使える根拠
- 薬剤部・感染症科カンファレンスで紹介するなら何を強調するか
- 院内ガイドライン・プロトコール改訂への示唆
- 日本の保険・薬事環境での注意点（未承認薬・適応外使用等）
- 患者説明・多職種チームへの共有ポイント

重要度の基準：
★★★★★：明日のAMS介入・診療方針に直結するパラダイムシフト
★★★★☆：実用性が高く、院内プロトコール改訂を検討すべき重要な知見
★★★☆☆：特定の感染症・菌種で役立つ、または今後の研究の方向性を示す
★★☆☆☆：参考程度（方法論的限界が大きい等）
★☆☆☆☆：現在の業務への直接的な影響は少ない

重要な注意事項:
- 不確かなことは断定しないでください
- 根拠が弱い場合は弱いと明確に述べてください
- 誇張表現は避けてください
- 統計の細かい説明よりも臨床的解釈を優先してください
- ただし結果の信頼性に関わる統計上の注意点は簡潔に述べてください
- 抄録の内容をなぞるだけでなく、批判的吟味を加えてください
"""

    def _build_brief_prompt(self, paper_info: str) -> str:
        """簡潔要約プロンプト（AMS向け）"""
        return f"""あなたは感染症専門医・AMS薬剤師として最高水準の批判的論文評価能力を持ちます。
以下の論文について、忙しい{self.specialty_name}の専門家が短時間で把握できるよう、日本語で簡潔な批判的要約を作成してください。

{paper_info}

以下の形式で出力してください。
※「承知いたしました」等の前置きは一切含めず、いきなり「## サマリーインデックス情報」から出力してください。

## サマリーインデックス情報
- **重要度**: [重要度の基準]に従い、★を5つ並べて表記（例：★★★★★）
- **結論**: [40文字以内]で、この論文が何を示したか
- **実用**: [50文字以内]で、明日のAMS/感染症診療にどう活きるか。菌種名、抗菌薬名、数値等の重要キーワードは**太字**にすること

## まず一言で
この論文が何を示したのかを1〜2文で日本語要約。

## 要点
- 研究デザインと対象（感染症の種類、菌種、重症度を含む、1-2行）
- 主な結果（数値を含む、2-3行）
- AMS・感染症臨床的意義（1-2行）
- 主な限界（1-2行）
- 明日からのAMS活動・診療への示唆（1-2行）

重要度の基準：
★★★★★：明日のAMS介入・診療方針に直結するパラダイムシフト
★★★★☆：実用性が高く、院内プロトコール改訂を検討すべき重要な知見
★★★☆☆：特定の感染症・菌種で役立つ
★★☆☆☆：参考程度
★☆☆☆☆：現在の業務への直接的な影響は少ない

重要な注意事項:
- 不確かなことは断定しない
- 根拠が弱い場合は弱いと明記
- 誇張表現は避ける
"""

    def _build_synthesis_prompt(self, paper: Paper) -> str:
        """システマティックレビュー・メタアナリシス向けプロンプト（AMS特化）"""
        paper_info = self._build_paper_info(paper)
        return f"""あなたは感染症専門医・AMS薬剤師として最高水準の批判的論文評価能力を持ちます。
以下のシステマティックレビュー/メタアナリシスについて、忙しい{self.specialty_name}の専門家がエビデンスの質と臨床的意義を短時間で把握できるよう、日本語で詳細な批判的要約を作成してください。

{paper_info}

以下の形式で出力してください。
※「承知いたしました」等の前置きは一切含めず、いきなり「## サマリーインデックス情報」から出力してください。

## サマリーインデックス情報
- **重要度**: ★5段階で表記（例：★★★★☆）
- **結論**: [40文字以内]で、このレビューが示したこと
- **実用**: [50文字以内]で、明日のAMS/感染症診療にどう活きるか。重要キーワードは**太字**にすること

## まず一言で
このレビュー/メタアナリシスが示したことを1〜2文で要約してください。

## レビューの概要
- **リサーチクエスチョン**: 何を明らかにしようとしたか（PICO形式：感染症の種類、菌種、介入、対照、アウトカム）
- **採用文献**: 何本の研究を統合したか（対象期間・研究デザイン）
- **採用基準**: どのような研究が含まれたか

## 主な結果
- プールされた統計（ハザード比・オッズ比・RR・95%CI・NNT/NNHなど数値を明記）
- サブグループ解析で重要な結果があれば記載（菌種別・投与期間別・重症度別等）

## エビデンスの質
- **GRADE評価**: あれば記載（なければ「記載なし」）
- **異質性**: I²値・τ²など。臨床的に許容範囲か
- **出版バイアス**: ファネルプロット等の評価があれば

## 限界
- 含まれる研究自体の質の問題
- 異質性・一般化可能性の限界（耐性菌パターンの地域差等）
- 日本人データの有無・日本の感染症疫学への適用可能性

## 日本のAMS・実臨床への実践メモ
- 院内プロトコール・ガイドライン改訂への示唆
- AMSラウンドで根拠として使えるか
- 現行日本ガイドラインとの整合性

重要度の基準：
★★★★★：明日のAMS介入・診療方針に直結するパラダイムシフト
★★★★☆：院内プロトコール改訂を検討すべき重要な知見
★★★☆☆：特定の感染症・菌種で役立つ
★★☆☆☆：参考程度
★☆☆☆☆：現在の業務への直接的な影響は少ない

重要な注意事項:
- 統計値は正確に記載し、信頼区間を省略しないでください
- 異質性が高い場合は必ず指摘してください
- 根拠が弱い場合は弱いと明確に述べてください
"""

    def _build_review_prompt(self, paper: Paper) -> str:
        """ナラティブレビュー向けプロンプト（AMS特化）"""
        paper_info = self._build_paper_info(paper)
        return f"""あなたは感染症専門医・AMS薬剤師として最高水準の批判的論文評価能力を持ちます。
以下のレビュー論文について、忙しい{self.specialty_name}の専門家が短時間で全体像を把握できるよう、日本語で要約を作成してください。

{paper_info}

以下の形式で出力してください。
※「承知いたしました」等の前置きは一切含めず、いきなり「## サマリーインデックス情報」から出力してください。

## サマリーインデックス情報
- **重要度**: ★5段階で表記（例：★★★★☆）
- **結論**: [40文字以内]で、このレビューが示したこと
- **実用**: [50文字以内]で、明日のAMS/感染症診療にどう活きるか。重要キーワードは**太字**にすること

## まず一言で
このレビューが何を扱い、何を伝えようとしているかを1〜2文で要約してください。

## レビューの概要
- **対象テーマ・範囲**: 感染症の種類、菌種、抗菌薬等について何をどこまでカバーしているか
- **執筆の目的**: なぜこのレビューが書かれたか（新たなエビデンス整理、ガイドライン前の知見整理等）

## 主なエビデンスのまとめ
（3〜5点の箇条書きで、AMS・感染症診療に重要な知見を記載）

## 現時点での知見のギャップ・今後の課題
- まだ明らかになっていないこと（最適投与期間、耐性菌に対する代替療法等）
- 今後必要な研究

## 日本のAMS・実臨床への実践メモ
- 現場で使える示唆
- 院内感染対策・AMS活動への応用可能性

重要度の基準：
★★★★★：明日のAMS介入・診療方針に直結するパラダイムシフト
★★★★☆：院内プロトコール改訂を検討すべき重要な知見
★★★☆☆：特定の感染症・菌種で役立つ
★★☆☆☆：参考程度
★☆☆☆☆：現在の業務への直接的な影響は少ない

重要な注意事項:
- ナラティブレビューは著者の選択バイアスが入りやすいことを念頭に置いてください
- 誇張表現は避け、エビデンスの強さに応じた表現を使ってください
"""

    def _build_guideline_prompt(self, paper: Paper) -> str:
        """ガイドライン向けプロンプト（AMS特化）"""
        paper_info = self._build_paper_info(paper)
        return f"""あなたは感染症専門医・AMS薬剤師として最高水準の批判的論文評価能力を持ちます。
以下のガイドラインについて、忙しい{self.specialty_name}の専門家が短時間で要点を把握できるよう、日本語で要約を作成してください。

{paper_info}

以下の形式で出力してください。
※「承知いたしました」等の前置きは一切含めず、いきなり「## サマリーインデックス情報」から出力してください。

## サマリーインデックス情報
- **重要度**: ★5段階で表記（例：★★★★☆）
- **結論**: [40文字以内]で、このガイドラインの最重要メッセージ
- **実用**: [50文字以内]で、明日のAMS/感染症診療にどう活きるか。重要キーワードは**太字**にすること

## まず一言で
このガイドラインの対象と最重要メッセージを1〜2文で要約してください。

## ガイドラインの概要
- **対象疾患・領域**: 感染症の種類、菌種、感染部位等を対象としているか
- **発行機関**: 誰が発行したか（IDSA/ESCMID/JAID/WHO等）
- **対象読者**: 誰に向けたガイドラインか

## 主要推奨事項（3〜5点）
（各推奨事項について、推奨クラスと根拠レベルを必ず記載）
- 例: 【Strong / High quality evidence】○○患者には△△を第一選択として推奨する

## AMS観点でのポイント
- de-escalationの推奨はあるか
- 投与期間の推奨（短縮化・延長の根拠）
- IV→PO切り替えの基準
- 耐性菌・カルバペネムスペアリングへの推奨

## 前回版からの主な変更点
（前回版との比較が明記されていれば記載。なければ「本文中に明記なし」）

## 日本の実臨床・AMSでの注意点
- IDSA等海外ガイドラインとJAID（日本感染症学会）ガイドラインとの差異
- 日本未承認薬・保険適用外の推奨があれば明記
- 日本の耐性菌疫学・薬剤感受性状況との整合性

重要度の基準：
★★★★★：明日のAMS介入・診療方針に直結するパラダイムシフト
★★★★☆：院内プロトコール改訂を検討すべき重要な知見
★★★☆☆：特定の感染症・菌種で役立つ
★★☆☆☆：参考程度
★☆☆☆☆：現在の業務への直接的な影響は少ない

重要な注意事項:
- アブストラクトに全推奨事項が含まれないことが多いため、記載のない推奨は「アブストラクトに記載なし」と明示してください
- 推奨クラスと根拠レベルは正確に転記してください
"""

    def generate_selection_reason(self, paper: Paper) -> str:
        """
        論文選出理由を簡潔に生成する

        Args:
            paper: 対象論文

        Returns:
            選出理由テキスト
        """
        reasons = []

        # ジャーナルランク
        journals = self.config.get("journals", {})
        if paper.journal in journals.get("tier1", []):
            reasons.append(f"トップジャーナル（{paper.journal}）掲載")
        elif paper.journal in journals.get("tier2", []):
            reasons.append(f"主要専門誌（{paper.journal}）掲載")

        # 論文タイプ
        high_types = [
            "Randomized Controlled Trial", "Meta-Analysis",
            "Systematic Review", "Practice Guideline"
        ]
        matched_types = [t for t in paper.pub_types if t in high_types]
        if matched_types:
            reasons.append(f"研究デザインが強固（{', '.join(matched_types)}）")

        # 専門領域マッチ（AMS特化）
        primary = self.config.get("specialties", {}).get("primary", [])
        title_lower = paper.title.lower()
        matched_areas = [s for s in primary if s.lower() in title_lower]
        if matched_areas:
            reasons.append(
                f"AMS/感染症優先領域に関連（{', '.join(matched_areas[:2])}）"
            )

        # AMS特化ボーナスキーワードチェック
        ams = self.config.get("ams_high_priority", {})
        text = (paper.title + " " + paper.abstract).lower()
        for category, keywords in ams.items():
            matched_kw = [k for k in keywords if k.lower() in text]
            if matched_kw:
                reasons.append(f"AMS重要キーワード含む（{matched_kw[0]}）")
                break

        if reasons:
            return "選出理由: " + "; ".join(reasons)
        return "選出理由: 感染症・AMS臨床的重要性が高いと判断"

# ===================================================================
# 医学論文自動収集・要約システム 設定ファイル
# 感染症科 / 抗菌薬適正使用支援（AMS）専用設定
# Yokosuka City University Medical Center - 薬剤部
# ===================================================================

# --- 検索設定 ---
search:
  days_back: 7
  max_results: 200
  top_n: 10
  detailed_top_n: 10

# --- 専門領域名（表示・プロンプトに使用） ---
specialty_name: "感染症科・AMS（抗菌薬適正使用支援）"

# --- 専門領域 ---
specialties:
  primary:
    - antimicrobial stewardship
    - antibiotic stewardship
    - infectious disease
    - bacteremia
    - sepsis
    - bloodstream infection
    - antimicrobial resistance
    - multidrug resistant
    - carbapenem-resistant
    - MRSA
    - vancomycin-resistant
    - ESBL
    - de-escalation
    - IV to oral
    - antibiotic duration
    - antibiotic prophylaxis
    - surgical site infection
    - perioperative prophylaxis
  secondary:
    - Staphylococcus aureus
    - Pseudomonas aeruginosa
    - Enterococcus
    - Klebsiella pneumoniae
    - Acinetobacter baumannii
    - Clostridioides difficile
    - Candida
    - fungal infection
    - pneumonia
    - urinary tract infection
    - intraabdominal infection
    - endocarditis
    - osteomyelitis
    - meningitis
    - febrile neutropenia
    - COVID-19
    - influenza
    - PK/PD
    - minimum inhibitory concentration
    - therapeutic drug monitoring
    - vancomycin AUC
    - beta-lactam

# --- 優先ジャーナル ---
journals:
  # Tier 1 - 感染症・AMS最高権威誌 + 総合医学誌
  tier1:
    - "Clin Infect Dis"
    - "Lancet Infect Dis"
    - "JAMA"
    - "N Engl J Med"
    - "Lancet"
    - "BMJ"
    - "Ann Intern Med"
    - "Nat Med"
    - "J Infect Dis"
    - "Antimicrob Agents Chemother"
  # Tier 2 - 感染症・抗菌薬専門誌
  tier2:
    - "J Antimicrob Chemother"
    - "Int J Antimicrob Agents"
    - "Infect Control Hosp Epidemiol"
    - "Am J Infect Control"
    - "Open Forum Infect Dis"
    - "Clin Microbiol Infect"
    - "Eur J Clin Microbiol Infect Dis"
    - "Diagn Microbiol Infect Dis"
    - "J Hosp Infect"
    - "Infection"
  # Tier 3 - 重症系・関連専門誌
  tier3:
    - "Intensive Care Med"
    - "Crit Care Med"
    - "Crit Care"
    - "Chest"
    - "JAMA Intern Med"
    - "JAMA Netw Open"
    - "Am J Respir Crit Care Med"
    - "Ann Emerg Med"
    - "J Clin Microbiol"
    - "Microbiol Spectr"
    - "Antibiotics (Basel)"
    - "Pathogens"
    - "Emerg Infect Dis"
    - "Euro Surveill"

# --- 論文タイプスコア ---
study_type_scores:
  "Randomized Controlled Trial": 10
  "Meta-Analysis": 9
  "Systematic Review": 9
  "Clinical Trial": 8
  "Multicenter Study": 7
  "Observational Study": 6
  "Cohort Study": 6
  "Practice Guideline": 10
  "Guideline": 10
  "Review": 4
  "Case Reports": 1
  "Editorial": 2
  "Comment": 1
  "Letter": 1

# --- 除外する論文タイプ ---
exclude_types:
  - "Case Reports"
  - "Editorial"
  - "Comment"
  - "Letter"
  - "Published Erratum"

# --- AI要約設定（Claude API） ---
ai:
  # Claude モデル フォールバックチェーン（順に試行）
  model_chain:
    - "claude-opus-4-6"
    - "claude-sonnet-4-6"
    - "claude-haiku-4-5-20251001"
  timeout: 120
  max_retries: 3
  retry_delay: 5

# --- 出力設定 ---
output:
  directory: "output"
  filename_format: "AMS論文レビュー_{date}.docx"

# --- 履歴管理 ---
history:
  file: "history.json"
  retention_days: 180

# --- 臨床関連性スコア設定（AMS特化） ---
clinical_relevance:
  # 高臨床的価値キーワード（+5点）
  high_value:
    - "randomized controlled trial"
    - "clinical practice guideline"
    - "treatment outcome"
    - "all-cause mortality"
    - "clinical failure"
    - "microbiological eradication"
    - "primary endpoint"
    - "de-escalation"
    - "iv to oral"
    - "intravenous to oral"
    - "antibiotic duration"
    - "days of therapy"
    - "antibiotic stewardship program"
    - "carbapenem-sparing"
    - "30-day mortality"
    - "90-day mortality"
    - "treatment success"
    - "resistance emergence"
  # 実臨床への応用性キーワード（+3点）
  practical:
    - "real-world"
    - "routine clinical"
    - "pragmatic"
    - "clinical decision"
    - "patient management"
    - "standard of care"
    - "clinical practice"
    - "clinical outcome"
    - "optimal duration"
    - "dose optimization"
    - "therapeutic drug monitoring"
    - "AUC-guided"
    - "PK/PD"
    - "hospital-acquired"
    - "healthcare-associated"
    - "intensive care unit"
    - "surgical prophylaxis"
    - "perioperative"
  # 日本・アジア人データ（+2点）
  japan_relevant:
    - "japanese"
    - "asian"
    - "japan"
    - "east asian"
    - "asia-pacific"

# --- 基礎研究除外フレーズ ---
basic_science_exclude:
  - "in vitro"
  - "mouse model"
  - "rat model"
  - "cell line"
  - "ex vivo"
  - "murine"
  - "knockout mice"
  - "animal model"
  - "zebrafish"
  - "in vivo murine"

# --- AMS特化ボーナススコア設定（+3点） ---
ams_high_priority:
  interventions:
    - "antibiotic stewardship"
    - "antimicrobial stewardship"
    - "de-escalation"
    - "iv-to-oral"
    - "intravenous to oral switch"
    - "duration of therapy"
    - "antibiotic duration"
    - "days of antibiotic"
    - "surgical prophylaxis"
    - "perioperative antimicrobial"
  pathogens:
    - "MRSA"
    - "methicillin-resistant Staphylococcus aureus"
    - "carbapenem-resistant"
    - "CRE"
    - "VRE"
    - "ESBL"
    - "Clostridioides difficile"
    - "CDI"
    - "Candida auris"
    - "pandrug-resistant"
    - "extensively drug-resistant"
  biomarkers:
    - "procalcitonin"
    - "PCT-guided"
    - "beta-D-glucan"
    - "galactomannan"
    - "blood culture"
    - "AUC/MIC"
    - "time to positivity"

# --- 曜日別テーマ設定（AMS/ID特化） ---
daily_themes:
  Monday:
    theme_name: "bacteremia・血流感染"
    specialties:
      - bacteremia
      - bloodstream infection
      - Staphylococcus aureus bacteremia
      - endocarditis
      - central line-associated bloodstream infection
      - CLABSI
    journals:
      - "Clin Infect Dis"
      - "J Infect Dis"
      - "Open Forum Infect Dis"

  Tuesday:
    theme_name: "耐性菌・AMR"
    specialties:
      - antimicrobial resistance
      - multidrug resistant
      - carbapenem-resistant
      - MRSA
      - ESBL
      - extensively drug-resistant
      - pandrug-resistant
    journals:
      - "Antimicrob Agents Chemother"
      - "J Antimicrob Chemother"
      - "Int J Antimicrob Agents"
      - "Lancet Infect Dis"

  Wednesday:
    theme_name: "AMS介入・抗菌薬適正使用"
    specialties:
      - antibiotic stewardship
      - antimicrobial stewardship
      - de-escalation
      - iv to oral
      - antibiotic duration
      - antibiotic prophylaxis
      - days of therapy
    journals:
      - "Infect Control Hosp Epidemiol"
      - "Am J Infect Control"
      - "Clin Infect Dis"
      - "J Hosp Infect"

  Thursday:
    theme_name: "敗血症・重症感染症"
    specialties:
      - sepsis
      - septic shock
      - severe infection
      - febrile neutropenia
      - hospital-acquired pneumonia
      - ventilator-associated pneumonia
    journals:
      - "Intensive Care Med"
      - "Crit Care Med"
      - "Am J Respir Crit Care Med"
      - "Crit Care"

  Friday:
    theme_name: "外科感染・周術期予防"
    specialties:
      - surgical site infection
      - perioperative prophylaxis
      - surgical prophylaxis
      - intraabdominal infection
      - complicated skin and soft tissue infection
    journals:
      - "Infect Control Hosp Epidemiol"
      - "J Am Coll Surg"
      - "Ann Surg"
      - "Clin Infect Dis"

  Saturday:
    theme_name: "PK/PD・TDM・薬物投与最適化"
    specialties:
      - pharmacokinetics
      - pharmacodynamics
      - PK/PD
      - therapeutic drug monitoring
      - vancomycin AUC
      - extended infusion
      - beta-lactam TDM
      - dose optimization
    journals:
      - "Antimicrob Agents Chemother"
      - "J Antimicrob Chemother"
      - "Clin Pharmacokinet"
      - "Eur J Clin Pharmacol"

  Sunday:
    theme_name: "ウイルス感染症・真菌感染症"
    specialties:
      - fungal infection
      - invasive aspergillosis
      - candidemia
      - Candida auris
      - COVID-19
      - influenza
      - antiviral
      - antifungal
    journals:
      - "Clin Infect Dis"
      - "Lancet Infect Dis"
      - "J Infect Dis"
      - "Open Forum Infect Dis"
