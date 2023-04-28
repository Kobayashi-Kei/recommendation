# ファイル一覧
- data_shaping
    - scicite
        - collectScicitePaper.py... Sciciteのデータセットに含まれる論文のアブストラクトをSemantic SchlorのAPIから取得する。
        - shapeDataForSSC.py... collectScicitePaper.pyで取得したアブストラクトをSSCで分類するために整形する
- ルート
    - analyzeWithScicite.py...
    - embedding.py...タイトル、アブスト全体をBERT・SciBERT・SPECTERで埋め込み、ファイル出力する。実行する際にはcondaのscibert環境に切り替えて実行する。
    - lineNotifier.py... プログラムからLINEに通知を飛ばすことができる。
    - run-labeled-20211106.py...
    - run-labeled-all.py... 論文推薦の評価プログラム。分割・埋め込み済みのアブストをもとに全クラス同士で比較する（例. 背景-背景,背景-目的, ..., 結果-結果）
    - run-labeled-max.py... 論文推薦の評価プログラム。分割・埋め込み済みのアブストをもとに同クラス同士で比較して最大を採用する。
    - run-labeled.py... 論文推薦の評価プログラム。分割・埋め込み済みのアブストをもとに同クラス同士で比較して和を採用する。
    - run.py... 論文推薦の評価プログラム。埋め込み済みのアブスト全体をもとに比較する。
    - shape-and-embedding.py... SSCの結果ファイルと論文データのjsonファイルを読み込んで、アブストラクトの分類結果を辞書形式で格納する。そしてそれらの文をBERT・SciBERT・SPECTERで埋め込む。
    - shapeLabeledAbstract.py... shape-and-emmbedding.pyのBERT系の埋め込みなしのデータの整形のみのバージョン。
    - test.py... pythonの動作確認のために自由に使ってるファイル。
    - tools.py... 汎用関数

    - visualizeDataset.py...データセットの被引用数・平均、分散を可視化する
    - visualizeLabel.py...
    - visualDist.py...ターゲット論文と被引用論文との距離（類似度）をターゲット論文数分計算し、ヒストグラムで可視化する
    - visualDist-notCitationRelation.py... ターゲット論文とランダムに選んだ論文との距離（類似度）を数百ペア計算し、ヒストグラムで可視化する
    - visualDist-lockPaper.py... ターゲット論文を固定して被引用論文・引用関係に無い論文複数件とのターゲット論文との距離を１ターゲット論文につき1枚の図に可視化する
    - visualDist-3dim.py... 引用関係にあるもの同士と無いもの同士で3観点選んで3次元空間にプロットして可視化する
    
    

## run-labeledについて
論文推薦（類似度に基づくランキング）の実行、評価プログラムである。コサイン類似度（またはユークリッド距離）を類似度スコアとして大きいものから並び替えてランク付けする。
引数として、bow,tf-idf, bert, scibert, specterなどを受け取りそれに沿った