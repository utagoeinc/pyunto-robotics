# Pyunto Robotics

Pyunto の日記アプリからロボットにメッセージを送り、動く様子を見られます。

```bash
pip install pyunto-robotics
pyunto-robotics showqr
```

ターミナルに表示された QR コードを Pyunto アプリで読み取ると、カーポートに停まった、背中にソーラーパネルを載せたロボットのウィンドウが開きます。スマート
フォンの日記に「日光が当たる場所まで移動して、電力を取得してきて」と書くと、ロボットは外へ出て、
パネルが受けている電力を測りながら日なたを探し、充電して戻り、集めた電気で家の照明を点けます。

ロボットは**あなたのコンピュータ**で動きます。日記はエンドツーエンドで暗号化されていて、
復号はあなたのマシン上だけで行われます。Pyunto のサーバは部屋もカメラも見ていません。

---

## クイックスタート

コマンドは 3 つ、覚えるのは最初の 1 つだけです。

```bash
pip install pyunto-robotics
python scripts/download_model.py     # 言葉を読めるようにする（初回のみ・約 5.5 GB）
pyunto-robotics showqr               # ターミナルに QR コードが出ます
```

その QR を Pyunto アプリで読み取ってください。どの日記に入れるか、誰が動かしているかがアプリに
表示され、承認すると**そのままロボットが起動します**。2 つ目のコマンドも、ターミナルに何かを
貼り戻す作業も要りません。

```
waiting for the scan… (Ctrl-C to stop)
paired — opening the robot.

listening — message the robot from the Pyunto app. Ctrl-C to stop.
```

あとは日記に、**普通の言葉で**書くだけです。

```
日光が当たる場所まで移動して、電力を取得してきて
```

ロボットはどう理解したかを伝えてから外へ出て、測りながら日なたを探し、充電して戻り、家の照明を
点けます。途中の経過も同じスレッドに報告します。

ペアリング後、そのスペースをアプリで一度開いてください。日記はエンドツーエンド暗号化なので、
サーバではなくメンバーが鍵を渡す必要があります。

### 6 文字のコードが表示された場合

古いビルドのアプリでは QR ではなくコードが出ます。その場合はこちらです。

```bash
pyunto-robotics demo --pair K3F9QZ
```

### 機体を選ぶ

```bash
pyunto-robotics robots                  # 入っている機体の一覧
pyunto-robotics showqr --robot pet      # 特定の機体でペアリングして起動
pyunto-robotics demo --robot watch      # ペアリング済みなら
pyunto-robotics whoami                  # このロボットのアカウントと参加スペース
```

---

## 普通の言葉で書く

コマンドを覚える必要はありません。ローカルの言語モデルが文を読みます。

```
「そろそろ電気が足りないかも」        ->  find_sun -> goto(park)
「猫はどこ？」                      ->  find
「母の様子はどう？」                 ->  check
```

どれもキーワード表には載っていません。モデルはあなたのマシンで動くので、日記が理解のために
どこかへ送られることはありません。

Apple シリコンが必要です。それ以外の環境や、`download_model.py` を実行する前は、起動時に 1 行
断ったうえでキーワード照合に切り替わります（起動を拒否することはありません）。強制する場合は
`--no-llm` を付けてください。

---

## 自分のロボットをつなぐ

この SDK の本体は 6 機種ではなく、**どんなロボットでも**日記につなげる仕組みです。
実装するのは 1 クラス 1 メソッドだけです。

```python
from pyunto_robotics.api import SkillResult

class MyRobot:
    def run(self, action, argument=None, where=None, expect=None) -> SkillResult:
        if action == "goto":
            ok = my_control_stack.move_to(argument)
            return SkillResult(ok, f"{argument} まで行きました。" if ok
                                   else f"{argument} まで行けませんでした。")
        return SkillResult(False, f"'{action}' はまだできません。")
```

これがすべてです。暗号化された通信、スペースへの参加、メッセージの受け取り、指示の解釈、返信は
こちらが持ちます。実機でも、別のシミュレータ（Newton、Isaac、Gazebo）でも、HTTP API だけの
ロボットでも同じようにつながります。

動く例は [`examples/my_robot.py`](examples/my_robot.py)（約 40 行）にあります。ナビゲーションや
ドア開けなど**こちらの skills を自分の機体で再利用する**ための `RobotBody` も含めた完全な契約は
[`pyunto_robotics/api.py`](pyunto_robotics/api.py) に 1 ファイルでまとまっています。

自分のロボットを配布可能なパッケージにする場合は、エントリポイントを宣言します。

```toml
[project.entry-points."pyunto_robotics.robots"]
acme = "acme_robot:setup"
```

`pip install acme-robot` のあと、`pyunto-robotics demo --robot acme` がそのまま動きます。

---

## 動作環境

- macOS（Apple シリコン）。Windows と Linux は未検証です
- Python 3.11 以上
- Pyunto アプリと、ロボットを招待するプレミアムスペース

macOS ではシミュレータのウィンドウを `mjpython` が持つ必要がありますが、`pyunto-robotics` が
自動で切り替えるので、上のコマンドをそのまま入力すれば動きます。

任意の追加機能:

```bash
pip install 'pyunto-robotics[llm]'   # 端末内の言語モデルで指示を解釈（Apple シリコン）
```

無くても、日本語と英語のルール照合で上記の例は動きます。モデルのダウンロードは不要です。

---

## 何が守られ、何が守られないか

- 日記はエンドツーエンドで暗号化されています。復号は**ロボットを動かしているコンピュータ**の
  上だけで行われ、Pyunto のサーバは読めません。
- そのコンピュータを管理している人は、そのスペースに書かれたことをすべて読めます。招待時の
  ダイアログと、日記に残るお知らせの両方でメンバー全員に伝えています。
- ロボットが読めるのは、招待されたスペースだけです。
- アカウントと鍵は `~/.pyunto-robot` にあります。消すと別のロボットになり、招待し直しが必要です。

## ライセンス

Pyunto のロボット・シーン・SDK コードは当社のものです。MuJoCo（Apache-2.0）は依存関係です。
`scripts/fetch_asimov.py` が取得する Asimov-1 モデルは第三者のもの（CERN-OHL-S-2.0）で、この
パッケージには同梱していません。
