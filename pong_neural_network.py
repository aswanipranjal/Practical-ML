# Deep Q-learning (reinforcement learning)
import pygame
import random

# framerate
fps = 60

# Size of window
window_height = 400
window_width = 400

# Size of paddle
paddle_width = 10
paddle_height = 60

# Size of ball
ball_width = 10
ball_height = 10

# Speed
paddle_speed = 2
ball_x_speed = 3
ball_y_speed = 2

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Initialize
screen = pygame.display.set_mode(window_width, window_height)

def draw_ball(ball_x_pos, ball_y_pos):
	ball = pygame.rect(ball_x_pos, ball_y_pos, ball_width, ball_height)
	pygame.draw.rect(screen, white, ball)

def draw_paddle1(paddle1_y_pos):
	paddle1 = pygame.rect(paddle_buffer, paddle1_y_pos, paddle_width, paddle_height)
	pygame.draw.rect(screen, white, paddle1)

def draw_paddle2(paddle2_y_pos):
	paddle2 = pygame.rect(window_width - paddle_buffer - paddle_width, paddle2_y_pos, paddle_width, paddle_height)
	pygame.draw.rect(screen, white, paddle2)