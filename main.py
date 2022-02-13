import gym, time
import gym_anytrading

# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import A2C
from gym_anytrading.envs import StocksEnv
from finta import TA
import supersuit as ss
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('GME.csv')
df.dtypes
df.sort_values('Date', ascending=True, inplace=True)
df.head()
df.set_index('Date', inplace=True)
df.head()
#env = gym.make('stocks-v0', df=df, frame_bound=(5,250), window_size=5)



def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, df=df, frame_bound=(5,250), window_size=5)
        env.signal_features
        env.action_space
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

#Setting Signals
df['Volume'] = df['Volume'].apply(lambda x: float(x.replace(",", "")))
df.dtypes
df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)

df["SMA"] = TA.SMA(df)
df["SMM"] = TA.SMM(df)
df["SSMA"] = TA.SSMA(df)
df["EMA"] = TA.EMA(df)
df["DEMA"] = TA.DEMA(df)
df["TEMA"] = TA.TEMA(df)
df["TRIMA"] = TA.TRIMA(df)
df["TRIX"] = TA.TRIX(df)
df["VAMA"] = TA.VAMA(df)
df["ER"] = TA.ER(df)
df["KAMA"] = TA.KAMA(df)
df["ZLEMA"] = TA.ZLEMA(df)
df["WMA"] = TA.WMA(df)
df["HMA"] = TA.HMA(df)
df["EVWMA"] = TA.EVWMA(df)
df["VWAP"] = TA.VWAP(df)
df["SMMA"] = TA.SMMA(df)
df["FRAMA"] = TA.FRAMA(df)
df["MOM"] = TA.MOM(df)
df["ROC"] = TA.ROC(df)
df["RSI"] = TA.RSI(df)
df["IFT_RSI"] = TA.IFT_RSI(df)
df["TR"] = TA.TR(df)
df["ATR"] = TA.ATR(df)
df["SAR"] = TA.SAR(df)
df["BBWIDTH"] = TA.BBWIDTH(df)
df["PERCENT_B"] = TA.PERCENT_B(df)
df["ADX"] = TA.ADX(df)
df["STOCH"] = TA.STOCH(df)
df["STOCHD"] = TA.STOCHD(df)
df["STOCHRSI"] = TA.STOCHRSI(df)
df["WILLIAMS"] = TA.WILLIAMS(df)
df["UO"] = TA.UO(df)
df["AO"] = TA.AO(df)
df["MI"] = TA.MI(df)
df["TP"] = TA.TP(df)
df["ADL"] = TA.ADL(df)
df["CHAIKIN"] = TA.CHAIKIN(df)
df["MFI"] = TA.MFI(df)
df["OBV"] = TA.OBV(df)
df["WOBV"] = TA.WOBV(df)
df["VZO"] = TA.VZO(df)
df["PZO"] = TA.PZO(df)
df["EFI"] = TA.EFI(df)
df["CFI"] = TA.CFI(df)
df["EMV"] = TA.EMV(df)
df["CCI"] = TA.CCI(df)
df["COPP"] = TA.COPP(df)
df["CMO"] = TA.CMO(df)
df["QSTICK"] = TA.QSTICK(df)
df["FISH"] = TA.FISH(df)
df["SQZMI"] = TA.SQZMI(df)
df["VPT"] = TA.VPT(df)
df["FVE"] = TA.FVE(df)
df["VFI"] = TA.VFI(df)
df["MSD"] = TA.MSD(df)
df["STC"] = TA.STC(df)
df.fillna(0, inplace=True)
df.head(15)

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = add_signals
    
num_cpu = 100
env_id = "CartPole-v1"
env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12,50))
env2.signal_features
df.head()
env_maker = lambda: env2
env = DummyVecEnv([env_maker])
start = time.time()
if __name__ == '__main__':
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
model = A2C.load("Trained",env, verbose=1)

model.learn(total_timesteps=100)
print("Learn")
model.save("Trained")
env = MyCustomEnv(df=df, window_size=12, frame_bound=(80,250))
obs = env.reset()

end = time.time()

timetaken = end - start
print(f"Took {timetaken} seconds")

while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break
    
model.save("Trained")
print("Trained")