# CLAUDE.md

このファイルはこのリポジトリで作業するAIアシスタント（Claude Code等）向けのガイドです。コードベース構造、実行フロー、主要な規約をまとめます。

## プロジェクト概要

PubMed APIから医学論文を毎日自動収集し、Claude APIで日本語要約を生成、Word文書化してGmailで配信するパイプライン。GitHub Actionsで完全自動運用される（PC・サーバー不要）。本番設定は感染症科・AMS（抗菌薬適正使用支援）特化だが、`setup.py` で他の専門科にも切り替え可能。

## アーキテクチャ

```
main.py (orchestrator)
  ├─ pubmed_searcher.py   PubMed検索 → list[Paper]
  ├─ paper_filter.py      重複排除・スコアリング・上位N件選出
  ├─ ai_summarizer.py     Claude APIでの日本語要約（モデルチェーン）
  └─ word_generator.py    python-docxでWord文書を生成
```

5ステップの直列パイプライン。`Paper` データクラス（`pubmed_searcher.py:20`）が全モジュールを貫通する共有データモデル。

### モジュール責務
| ファイル | 責務 |
|---------|------|
| `main.py` | エントリポイント。設定ロード、曜日別テーマ適用、各モジュール起動 |
| `pubmed_searcher.py` | NCBI E-utilities (Biopython Entrez)。esearch→efetchバッチ取得。HTTP 400対策のフォールバックチェーン実装済み |
| `paper_filter.py` | スコアリング（論文タイプ + ジャーナル + 専門領域マッチ + 臨床関連性 + 最新性）。`history.json` で報告済み除外 |
| `ai_summarizer.py` | Claude APIラッパ。論文タイプ（guideline/synthesis/review/research）を自動判定し別プロンプトを使用。モデルフォールバックチェーン対応 |
| `word_generator.py` | Word文書生成。AI要約のMarkdown（`##`見出し、`**太字**`、`-`リスト）をdocxに変換。冒頭サマリーインデックス + 各論文詳細 + 末尾サマリーテーブル + 参考文献 |
| `setup.py` | 初期設定スクリプト。Gemini APIで専門領域に応じた `config.yaml` を自動生成 |
| `config.yaml` | 全モジュールの単一の設定ソース |
| `history.json` | 報告済みPMIDの記録（`retention_days` 経過後に自動削除） |

## 開発ワークフロー

### ローカル実行
```bash
# 依存関係（Python 3.11推奨）
pip install -r requirements.txt

# .env を作成し ANTHROPIC_API_KEY、NCBI_EMAIL、NCBI_API_KEY を設定
cp .env.example .env

# 通常実行（output/ にWord出力）
python main.py

# 検索期間を変える
python main.py --weeks-back 2

# AI要約とWord生成をスキップ（PubMed検索とフィルタのみ確認）
python main.py --dry-run

# 出力先指定
python main.py --output-dir ./tmp
```

`main.py` は冒頭で `os.chdir(script_dir)` するため、どこから起動してもリポジトリルートが作業ディレクトリになる。

### GitHub Actions（本番運用）
- `.github/workflows/daily_summary.yml`: 毎日 UTC 22:00（JST 翌7:00）に自動実行 + 手動実行可。実行後 `history.json` を自動コミットする。
- `.github/workflows/setup.yml`: 専門領域の手動セットアップ用。Gemini APIで `config.yaml` を生成・コミット。

### 必要なGitHub Secrets
| Secret | 用途 | 必須 |
|--------|------|------|
| `ANTHROPIC_API_KEY` | Claude API（要約生成） | ◯ |
| `GEMINI_API_KEY` | Gemini API（setup.pyのみ） | setup時のみ |
| `NCBI_EMAIL` | E-utilities識別 | ◯ |
| `NCBI_API_KEY` | レート制限緩和 (3→10 req/s) | ◯（推奨） |
| `GMAIL_USERNAME` / `GMAIL_APP_PASSWORD` / `GMAIL_RECIPIENT` | メール送信 | ◯ |

> 注意: README.md の旧説明では Gemini を要約に使うかのような記述があるが、現状の本番要約は Claude API（`ai_summarizer.py`）。Gemini は `setup.py` の config 生成専用。

## 設定（config.yaml）の主要セクション

| キー | 役割 |
|------|------|
| `search.days_back` | 検索対象期間（日数） |
| `search.max_results` / `top_n` / `detailed_top_n` | 取得上限・選出上限・詳細要約上限 |
| `specialty_name` | 表示・プロンプトに埋め込む領域名 |
| `specialties.primary` / `secondary` | PubMed `[Title/Abstract]` 検索キーワード（**全て小文字、論文に実際出現するフレーズ**） |
| `journals.tier1` / `tier2` / `tier3` | スコア重み（10/8/6点）。**ISO略称表記** |
| `study_type_scores` | 論文タイプ別スコア |
| `exclude_types` | 完全除外する論文タイプ |
| `ai.model_chain` | Claudeモデルのフォールバック順 |
| `clinical_relevance` | 臨床的価値スコア用キーワード |
| `basic_science_exclude` | 基礎研究除外フレーズ |
| `daily_themes.{Monday..Sunday}` | 曜日ごとに `specialties.primary` と `journals.tier1` を上書きするテーマ |

### 曜日別テーマの動作（`main.py:58-77`）
実行日の曜日（`datetime.now().strftime("%A")`）が `daily_themes` に存在すれば、その日の `specialties` と優先ジャーナルでconfigを **実行時に上書き**する（ファイルは書き換えない）。同時に `top_n=5`、`detailed_top_n=5` に固定される。

## スコアリング設計（`paper_filter.py:167-201`）

合計最大 ~48点：
- **論文タイプ**: 0–10（`study_type_scores`）
- **ジャーナル**: 10/8/6/3（tier1/2/3/それ以外）
- **専門領域マッチ**: 最大15（primary 5点×n, secondary 2点×n, タイトル+abstract+MeSH+keywordから）
- **臨床関連性**: 最大10（high_value 5 + practical 3 + japan_relevant 2）
- **最新性ボーナス**: 0–3（≤3日:3, ≤7日:2, ≤14日:1）

除外: `history.json` 既報、`exclude_types` のみのもの、abstract空、基礎研究のみ（`basic_science_exclude` 該当 かつ `clinical_words` 非該当）。

## AI要約プロンプト設計

`AISummarizer._detect_paper_type` が論文を4種に分類し、別々のプロンプトを使用：
- `guideline` — Practice Guideline / Guideline
- `synthesis` — Systematic Review / Meta-Analysis
- `review` — Review
- `research` — その他（`detailed=True` で詳細プロンプト、それ以外は簡潔プロンプト）

全プロンプトは `## サマリーインデックス情報` セクションから始まる構造で、`重要度` (★5段階)、`結論` (40文字以内)、`実用` (50文字以内, **太字**強調) を必ず含む。`word_generator._extract_index_info` が正規表現でこの構造を解析するため、**プロンプト変更時は出力フォーマットの不変条件を維持すること**。

### モデルフォールバックチェーン（`ai_summarizer.py:45-128`）
- レート制限・404・400・504・529 → 即座に次モデルへ
- それ以外のエラー → `max_retries` 回リトライ後に次モデルへ
- 現在のチェーン（`config.yaml`）: `claude-opus-4-6` → `claude-sonnet-4-6` → `claude-haiku-4-5-20251001`

## Word出力の構造（`word_generator.py`）

1. ヘッダ（タイトル + 専門領域名 + 作成日）
2. **サマリーインデックス**（全論文の重要度・結論・実用を一覧）
3. 各論文セクション（基本情報テーブル + 選出理由 + AI要約Markdown→Word変換）
4. **今週の論文一覧**テーブル（5列: 優先度・論文・ジャーナル/デザイン・一言要約・実臨床への影響）
5. **参考文献**

サポートするMarkdown構文: `## 見出し2`, `### 見出し3`, `- リスト`, `* リスト`, `**太字**`。それ以外（コードブロック、リンク等）はサポート外。

## 重要な規約・落とし穴

- **依存関係**: `requirements.txt` は本番（main.py）専用で `anthropic` を含む。`setup.py` は `google-genai` + `pyyaml` のみ必要で、setup workflow 内で個別インストールされる。
- **NCBI APIキーのバリデーション**: `pubmed_searcher.py:53-62` で長さ10超かつ `"none"` 以外でないと無効扱い。誤った値が設定されている場合、初回失敗時に自動的にAPIキーなしへフォールバックする。
- **PubMed HTTP 400対策**: `_execute_esearch` は3段階フォールバック（通常 → APIキー除外 → クエリ簡略化）。クエリ調整時はジャーナル名重複除去（`dict.fromkeys`）を維持すること。
- **キーワード形式**: `specialties.primary` 等は **小文字英語のフレーズ** で、論文Title/Abstractに実際に出現するもの。MeSHカテゴリ名（"Cardiovascular diseases"等）は使用禁止（`setup.py` のプロンプトで強調されている）。
- **ジャーナル名表記**: PubMed ISOAbbreviation（例: `"Clin Infect Dis"`、`"N Engl J Med"`）。フルネームでは一致しない。
- **コミットメッセージ**: 日本語で簡潔に。`fix:`, `setup:`, `chore:` 等のプレフィックスを使用する慣習。GitHub Actionsの自動コミットは `[skip ci]` を含めて再実行を抑止。
- **history.json の自動コミット**: 本番ワークフローが `git pull --rebase --autostash` 後に push する。ローカルで編集した変更が同期されないと衝突するため、ローカル開発後は pull してから作業すること。
- **タイムアウト**: ワークフローは 30分。要約対象が増えると上限に近づくため、`detailed_top_n` を増やす場合は注意。

## ファイル構成早見

```
.
├── .env.example            ローカル開発用の環境変数テンプレ
├── .github/workflows/
│   ├── daily_summary.yml   毎朝の本番ジョブ（cron + 手動）
│   └── setup.yml           config.yaml生成ジョブ（手動）
├── ai_summarizer.py        Claude API要約
├── config.yaml             全設定（現状: 感染症・AMS特化）
├── history.json            報告済みPMID（自動更新）
├── main.py                 エントリポイント
├── paper_filter.py         スコアリング・選出
├── pubmed_searcher.py      PubMed取得 + Paperデータクラス
├── README.md               エンドユーザー向けセットアップ手順
├── requirements.txt        本番依存（biopython, python-docx, anthropic, pyyaml, python-dotenv）
├── setup.py                Gemini APIによるconfig生成
└── word_generator.py       Word文書生成
```

## 変更時に意識すべきこと

- `Paper` データクラスのフィールドを変えると、検索→フィルタ→要約→Wordの全段階に影響する。
- AI要約プロンプトの `## サマリーインデックス情報` セクション形式を変更すると、`word_generator._extract_index_info` の正規表現が破綻しサマリーインデックスが空になる。
- `config.yaml` のキーを追加・変更したら、`setup.py` の `BASE_CONFIG` も合わせて更新すること（そうしないとSetupワークフロー再実行時に新キーが消える）。
- ロギングは `logging` 標準モジュール、出力は stdout + `paper_collector.log`。`logger.info` は日本語OK（READMEと整合）。
