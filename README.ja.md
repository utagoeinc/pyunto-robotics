# Pyunto Robotics

Pyunto の日記アプリからロボットにメッセージを送り、動く様子を見られます。

```bash
pip install pyunto-robotics
pyunto-robotics demo --pair K3F9QZ
```

オフィスに立つヒューマノイドのウィンドウが開きます。スマートフォンの日記に「ドアまで歩いて」と
書くと、ロボットがドアまで歩き、何をしたかを返事します。

ロボットは**あなたのコンピュータ**で動きます。日記はエンドツーエンドで暗号化されていて、
復号はあなたのマシン上だけで行われます。Pyunto のサーバは部屋もカメラも見ていません。

---

## デモの手順

1. Pyunto アプリでプレミアムスペースを開き、「ロボットを招待」を選びます。6 文字のペアリング
   コードが表示されます。
2. お使いのコンピュータで実行します。
   ```bash
   pyunto-robotics demo --pair <コード>
   ```
3. そのスペースをアプリで一度開いてください。日記はエンドツーエンド暗号化なので、サーバが鍵を
   渡すことはできず、メンバーが鍵を配る必要があります。準備ができるとロボットが挨拶します。
4. 日記を書くと、ロボットが動いて同じスレッドに返事をします。

同梱のロボットとワールドは 4 種類です。

| `--robot` | 機体 | ワールド |
|---|---|---|
| `office`（既定） | H1 ヒューマノイド | 3 部屋のオフィス。ドアを開ける |
| `home` | Momo ヒューマノイド | 洗濯室。洗濯機から出して畳む |
| `patrol` | Q1 四足歩行 | 屋外。建物の周囲を巡回し階段を登る |
| `lunar` | R1 ローバー | 月の南極。クレーターの地面を走る |

```bash
pyunto-robotics robots      # 入っている機体の一覧
pyunto-robotics whoami      # このロボットのアカウントと参加スペース
```

---

## 自分のロボットをつなぐ

この SDK の本体は 4 機種ではなく、**どんなロボットでも**日記につなげる仕組みです。
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
