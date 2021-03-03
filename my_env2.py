import gym
import numpy as np
import pyautogui
import time

from capture import Capture
from cell_check import Cnn

class Env(object):
    capture = Capture()
    cnn = Cnn()
    
    def __init__(self):
        """
        すべてのエージェントによって使用される抽象環境クラス。このクラスには正確な
        OpenAI Gymが使用するのと同じAPIであるため、OpenAIGymとの統合は簡単です。とは対照的に
        OpenAI Gymの実装、このクラスは実際のメソッドなしで抽象メソッドのみを定義します
        実装。
        独自の環境を実装するには、次のメソッドを定義する必要があります。
        -`ステップ `
        -`リセット `
        -`render`
        -`閉じる `
        [ジムのドキュメント]（https://gym.openai.com/docs/#environments）を参照してください。
        # """
        # reward_range = (-np.inf, np.inf)
        # action_space = None
        # observation_space = None
        self.ACTION = ['w','s','d','a','c','f']
        ACTION_NUM = 6
        self.action_space = gym.spaces.Discrete(ACTION_NUM)
        self.labels = ['cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell_close', 'cell_flag', 'cell_flag_miss', 'cell_open', 'cell_open_miss']


        self.capture.taking_pictures()
        self.before_senter_status = self.check_labels(self.cnn.check_one('./capture/_24.png'))[0]
        
        # 状態の範囲を定義
        # MAP = np.array([
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0]
        # ])

        MAP = np.array([
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
        ])
        self.observation_space = gym.spaces.Box(low=0, high=11, shape=MAP.shape )
        self.reset()

        # 報酬の幅
        self.reward_range = [-5., 10.]

    def step(self, action_lindex):
        """
        環境のダイナミクスの1つのタイムステップを実行します。
        アクションを受け入れ、タプル（監視、報酬、完了、情報）を返します。
        ＃引数
            アクション（オブジェクト）：環境によって提供されるアクション。
        ＃ 戻り値
            観察（オブジェクト）：エージェントによる現在の環境の観察。
            報酬（フロート）：前のアクションの後に返された報酬の量。
            done（boolean）：エピソードが終了したかどうか。この場合、さらにstep（）を呼び出すと、未定義の結果が返されます。
            info（dict）：補助的な診断情報が含まれています（デバッグに役立ち、場合によっては学習に役立ちます）。
        """
        # pyautogui.press
        action = self.ACTION[action_lindex]
        pyautogui.press(action)
        time.sleep(0.4)
        self.capture.taking_pictures()
        carent_senter_status = self.check_labels(self.cnn.check_one('./capture/_24.png'))[0]
        reward = 0

        # 空いているとき
        if self.before_senter_status in ['cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell_open']:
            if action in self.ACTION[:4]:
                if carent_senter_status == 'cell_flag_miss':
                    reward = 0.5
                else:
                    reward = -0.5
            elif action in self.ACTION[4:]:
                reward = -2
        
        # 空いていないとき
        elif self.before_senter_status == 'cell_close':
            if action == 'c':
                # 1～6の時
                if carent_senter_status in ['cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell_open']:
                    reward = 6
                # 開かなかったとき
                elif carent_senter_status == 'cell_close':
                    reward = -1.5
                # 間違ったとき
                elif carent_senter_status == 'cell_open_miss':
                    reward = -6
                else: #carent_senter_status in ['cell_flag', 'cell_flag_miss']:
                    print('open error')

            elif action == 'f':
                if carent_senter_status == 'cell_flag':
                    reward = 7
                elif carent_senter_status == 'cell_flag_miss':
                    reward = -3
                elif carent_senter_status == 'cell_close':
                    reward = -1.5
                else: #carent_senter_status in ['cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell_open', 'cell_open_miss']:
                    print('flag error')
            
            # 移動
            else:
                # 空いてないときに空いてない方に移動
                if carent_senter_status == 'cell_close':
                    reward = -1
                elif carent_senter_status == 'cell_flag_miss':
                    reward = 0
                else:
                    reward = -0.3
            
        # フラグミスのとき
        elif self.before_senter_status == 'cell_flag_miss':
            if action == 'c':
                reward = 2
            else:
                reward = -3

        elif self.before_senter_status in ['cell_flag', 'cell_open_miss']:
            if action == 'c':
                reward = -2
            if action in self.ACTION[:4]:
                if carent_senter_status == 'cell_flag_miss':
                    reward = 0
                else:
                    reward = -0.5
            elif action == 'f':
                reward = -7


        print('  : ',reward, " ", action, ' ',self.before_senter_status , ' ', carent_senter_status)
        # nan
        observation = self.cnn.check_55()
        observation = observation.reshape(5,5)
        done = False

        self.before_senter_status = carent_senter_status

        return observation, reward, done, {'episode_reward':reward}

    def reset(self):
        """
        環境の状態をリセットし、最初の観測を返します。
        ＃ 戻り値
            観測（オブジェクト）：空間の最初の観測。初期報酬は0と見なされます。
        """
        print('reset')
        observation = self.cnn.check_55()
        observation = observation.reshape(5,5)
        return observation

    def render(self, mode='human', close=False):
        """
        環境をレンダリングします。
        サポートされるモードのセットは、環境によって異なります。 （いくつかの
        環境はレンダリングをまったくサポートしていません。）
        ＃引数
            mode（str）：レンダリングに使用するモード。
            close（bool）：開いているすべてのレンダリングを閉じます。
        """
        print('render')

    def close(self):
        """
        サブクラスをオーバーライドして、必要なクリーンアップを実行します。
        環境は、次の場合に自動的に閉じます（）
        収集されたガベージまたはプログラムの終了時。
        """
        print('close')

    def seed(self, seed=None):
        """
        このenvの乱数ジェネレーターのシードを設定します。
        ＃ 戻り値
            この環境の乱数ジェネレーターで使用されるシードのリストを返します
        """
        print('seed')

    def configure(self, *args, **kwargs):
        """
        環境にランタイム構成を提供します。
        この構成は、
        環境の実行方法（リモートサーバーのアドレスなど、
        またはImageNetデータへのパス）。影響はありません
        環境のセマンティクス。
        """
        print('configure')

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    def check_labels(self, predict_classes):
        return [self.labels[class_num] for class_num in predict_classes]