#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 五子棋环境

import numpy as np
import pygame
from pygame.locals import *
import time

# 电脑字体的位置
FONT_PATH = 'C:/Windows/Fonts/simkai.ttf'

class Board(object):
    """棋盘游戏逻辑控制"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))  # 棋盘宽度
        self.height = int(kwargs.get('height', 15))  # 棋盘高度
        self.states = {}    # 棋盘状态为一个字典,键: 移动步数,值: 玩家的棋子类型
        self.n_in_row = int(kwargs.get('n_in_row', 5))  # 5个棋子一条线则获胜
        self.players = [1, 2]  # 玩家1,2

    def init_board(self, start_player=0):
        # 初始化棋盘

        # 当前棋盘的宽高小于5时,抛出异常(因为是五子棋)
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('棋盘的长宽不能少于{}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # 先手玩家
        self.availables = list(range(self.width * self.height)) # 初始化可用的位置列表
        self.states = {}  # 初始化棋盘状态
        self.last_move = -1  # 初始化最后一次的移动位置

    def move_to_location(self, move):
        # 根据传入的移动步数返回位置(如:move=2,计算得到坐标为[0,2],即表示在棋盘上左上角横向第三格位置)
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        # 根据传入的位置返回移动值
        # 位置信息必须包含2个值[h,w]
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        # 超出棋盘的值不存在
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        从当前玩家的角度返回棋盘状态。
    状态形式：4 * 宽 * 高
        """
        # 使用4个15x15的二值特征平面来描述当前的局面
        # 前两个平面分别表示当前player的棋子位置和对手player的棋子位置，有棋子的位置是1，没棋子的位置是0
        # 第三个平面表示对手player最近一步的落子位置，也就是整个平面只有一个位置是1，其余全部是0
        # 第四个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]   # 获取棋盘状态上属于当前玩家的所有移动值
            move_oppo = moves[players != self.current_player]   # 获取棋盘状态上属于对方玩家的所有移动值
            square_state[0][move_curr // self.width,            # 对第一个特征平面填充值(当前玩家)
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,            # 对第二个特征平面填充值(对方玩家)
                            move_oppo % self.height] = 1.0
            # 指出最后一个移动位置
            square_state[2][self.last_move // self.width,       # 对第三个特征平面填充值(对手最近一次的落子位置)
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:   # 对第四个特征平面填充值,当前玩家是先手,则填充全1,否则为全0
            square_state[3][:, :] = 1.0
        # 将每个平面棋盘状态按行逆序转换(第一行换到最后一行,第二行换到倒数第二行..)
        return square_state[:, ::-1, :]

    def do_move(self, move):
        # 根据移动的数据更新各参数
        self.states[move] = self.current_player  # 将当前的参数存入棋盘状态中
        self.availables.remove(move)  # 从可用的棋盘列表移除当前移动的位置
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )  # 改变当前玩家
        self.last_move = move  # 记录最后一次的移动位置

    def has_a_winner(self):
        # 是否产生赢家
        width = self.width  # 棋盘宽度
        height = self.height  # 棋盘高度
        states = self.states  # 状态
        n = self.n_in_row  # 获胜需要的棋子数量

        # 当前棋盘上所有的落子位置
        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            # 当前棋盘落子数在7个以上时会产生赢家,落子数低于7个时,直接返回没有赢家
            return False, -1

        # 遍历落子数
        for m in moved:
            h = m // width
            w = m % width  # 获得棋子的坐标
            player = states[m]  # 根据移动的点确认玩家

            # 判断各种赢棋的情况
            # 横向5个
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # 纵向5个
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            # 左上到右下斜向5个
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # 右上到左下斜向5个
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        # 当前都没有赢家,返回False
        return False, -1

    def game_end(self):
        """检查当前棋局是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            # 棋局布满,没有赢家
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

N = 9

IMAGE_PATH = 'UI/'

WIDTH = 540 # 棋盘图片宽
HEIGHT = 540    # 棋盘图片高
MARGIN = 22 # 图片上的棋盘边界有间隔
#GRID = (WIDTH - 2 * MARGIN) / (N - 1)   # 设置每个格子的大小
GRID = (270-22)/7
PIECE = 32  # 棋子的大小

# 加上UI的布局的训练方式
class Game_UI(object):
    """游戏控制区域"""

    def __init__(self, board, is_shown, **kwargs):
        self.board = board  # 加载棋盘控制类
        self.is_shown = is_shown

        # 初始化 pygame
        pygame.init()

        if is_shown != 0:
            self.__screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
            pygame.display.set_caption('五子棋AI')

            # UI 资源
            self.__ui_chessboard = pygame.image.load(IMAGE_PATH + 'chessboard.jpg').convert()
            self.__ui_piece_black = pygame.image.load(IMAGE_PATH + 'piece_black.png').convert_alpha()
            self.__ui_piece_white = pygame.image.load(IMAGE_PATH + 'piece_white.png').convert_alpha()

    # 将索引转换成坐标
    def coordinate_transform_map2pixel(self, i, j):
        # 从 逻辑坐标到 UI 上的绘制坐标的转换
        return MARGIN + j * GRID - PIECE / 2, MARGIN + i * GRID - PIECE / 2

    # 将坐标转换成索引
    def coordinate_transform_pixel2map(self, x, y):
        # 从 UI 上的绘制坐标到 逻辑坐标的转换
        i, j = int(round((y - MARGIN + PIECE / 2) / GRID)), int(round((x - MARGIN + PIECE / 2) / GRID))
        # 有MAGIN, 排除边缘位置导致 i,j 越界
        if i < 0 or i >= N or j < 0 or j >= N:
            return None, None
        else:
            return i, j

    def draw_chess(self):
        # 棋盘
        self.__screen.blit(self.__ui_chessboard, (0, 0))
        # 棋子
        for i in range(0, N):
            for j in range(0, N):
                # 计算移动位置
                loc = i * N + j
                p = self.board.states.get(loc, -1)

                player1, player2 = self.board.players

                # 求出落子的坐标
                x, y = self.coordinate_transform_map2pixel(i, j)

                if p == player1:  # 玩家为1时,将该位置放入黑棋
                    self.__screen.blit(self.__ui_piece_black, (x, y))
                elif p == player2:  # 玩家为2时,将该位置放入白棋
                    self.__screen.blit(self.__ui_piece_white, (x, y))
                else:
                    pass  # 当前位置无玩家落子时,跳过

    def one_step(self):
        i, j = None, None
        # 鼠标点击
        mouse_button = pygame.mouse.get_pressed()
        # 左键
        if mouse_button[0]:
            x, y = pygame.mouse.get_pos()
            i, j = self.coordinate_transform_pixel2map(x, y)

        if not i is None and not j is None:
            loc = i * N + j
            p = self.board.states.get(loc, -1)

            player1, player2 = self.board.players

            if p == player1 or p == player2:
                # 当前位置有棋子
                return False
            else:
                cp = self.board.current_player

                location = [i,j]
                move = self.board.location_to_move(location)
                self.board.do_move(move)

                if self.is_shown:
                    if cp == player1:
                        self.__screen.blit(self.__ui_piece_black, (x, y))
                    else:
                        self.__screen.blit(self.__ui_piece_white, (x, y))

                return True
        return False

    def draw_result(self, result):
        font = pygame.font.Font(FONT_PATH, 50)
        tips = u"本局结束:"

        player1, player2 = self.board.players

        if result == player1:
            tips = tips + u"玩家1胜利"
        elif result == player2:
            tips = tips + u"玩家2胜利"
        else:
            tips = tips + u"平局"
        text = font.render(tips, True, (255, 0, 0))
        self.__screen.blit(text, (WIDTH / 2 - 200, HEIGHT / 2 - 50))

    # 使用鼠标对弈(player1传入为人类玩家,player2为MCTS机器人)
    def start_play_mouse(self, player1, player2, start_player=0):
        """开始一局游戏"""
        if start_player not in (0, 1):
            # 如果玩家不在玩家1,玩家2之间,抛出异常
            raise Exception('开始的玩家必须为0(玩家1)或1(玩家2)')
        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = self.board.players  # 加载玩家1,玩家2
        player1.set_player_ind(p1)  # 设置玩家1
        player2.set_player_ind(p2)  # 设置玩家2
        players = {p1: player1, p2: player2}

        # 如果人类玩家不是先手
        if start_player != 0:
            current_player = self.board.current_player  # 获取当前玩家
            player_in_turn = players[current_player]  # 当前玩家的信息
            move = player_in_turn.get_action(self.board)  # 基于MCTS的AI下一步落子
            self.board.do_move(move)  # 根据下一步落子的状态更新棋盘各参数

        if self.is_shown:
            # 绘制棋盘
            self.draw_chess()
            # 刷新
            pygame.display.update()

        flag = False
        win = None

        while True:
            # 捕捉pygame事件
            for event in pygame.event.get():
                # 退出程序
                if event.type == QUIT:
                    pygame.quit()
                    exit()
                elif event.type == MOUSEBUTTONDOWN:
                    # 成功着棋
                    if self.one_step():
                        end, winner = self.board.game_end()
                    else:
                        continue
                    # 结束
                    if end:
                        flag = True
                        win = winner
                        break

                    # 没有结束,则使用MCTS进行下一步落子
                    current_player = self.board.current_player  # 获取当前玩家
                    player_in_turn = players[current_player]  # 当前玩家的信息

                    move = player_in_turn.get_action(self.board)  # 基于MCTS的AI下一步落子
                    self.board.do_move(move)  # 根据下一步落子的状态更新棋盘各参数

                    if self.is_shown:
                        # 展示棋盘
                        self.draw_chess()
                        # 刷新
                        pygame.display.update()

                    # 判断当前棋局是否结束
                    end, winner = self.board.game_end()
                    # 结束
                    if end:
                        flag = True
                        win = winner
                        break

            if flag and self.is_shown:
                self.draw_result(win)
                # 刷新
                pygame.display.update()
                break

    def start_play(self, player1, player2, start_player=0):
        """开始一局游戏"""
        if start_player not in (0, 1):
            # 如果玩家不在玩家1,玩家2之间,抛出异常
            raise Exception('开始的玩家必须为0(玩家1)或1(玩家2)')
        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = self.board.players  # 加载玩家1,玩家2
        player1.set_player_ind(p1)  # 设置玩家1
        player2.set_player_ind(p2)  # 设置玩家2
        players = {p1: player1, p2: player2}
        if self.is_shown:
            # 绘制棋盘
            self.draw_chess()
            # 刷新
            pygame.display.update()

        while True:
            if self.is_shown:
                # 捕捉pygame事件
                for event in pygame.event.get():
                    # 退出程序
                    if event.type == QUIT:
                        pygame.quit()
                        exit()

            current_player = self.board.current_player  # 获取当前玩家
            player_in_turn = players[current_player]  # 当前玩家的信息
            move = player_in_turn.get_action(self.board)  # 基于MCTS的AI下一步落子
            self.board.do_move(move)  # 根据下一步落子的状态更新棋盘各参数
            if self.is_shown:
                # 展示棋盘
                self.draw_chess()
                # 刷新
                pygame.display.update()

            # 判断当前棋局是否结束
            end, winner = self.board.game_end()
            # 结束
            if end:
                win = winner
                break
        if self.is_shown:
            self.draw_result(win)
            # 刷新
            pygame.display.update()
        return win

    def start_self_play(self, player, temp=1e-3):
        """ 使用MCTS玩家开始自己玩游戏,重新使用搜索树并存储自己玩游戏的数据
        (state, mcts_probs, z) 提供训练
        """
        self.board.init_board()  # 初始化棋盘
        states, mcts_probs, current_players = [], [], []  # 状态,mcts的行为概率,当前玩家

        if self.is_shown:
            # 绘制棋盘
            self.draw_chess()
            # 刷新
            pygame.display.update()

        while True:
            if self.is_shown:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

            # 根据当前棋盘状态返回可能得行为,及行为对应的概率
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # 存储数据
            states.append(self.board.current_state())  # 存储状态数据
            mcts_probs.append(move_probs)  # 存储行为概率数据
            current_players.append(self.board.current_player)  # 存储当前玩家
            # 执行一个移动
            self.board.do_move(move)
            if self.is_shown:
                # 绘制棋盘
                self.draw_chess()
                # 刷新
                pygame.display.update()

            # 判断该局游戏是否终止
            end, winner = self.board.game_end()
            if end:
                # 从每个状态的当时的玩家的角度看待赢家
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    # 没有赢家时
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MSCT的根节点
                player.reset_player()
                if self.is_shown:
                    self.draw_result(winner)

                    # 刷新
                    pygame.display.update()
                return winner, zip(states, mcts_probs, winners_z)
