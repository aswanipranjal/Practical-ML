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

def update_ball(paddle1_y_pos, paddle2_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction):
	# Update x and y positions
	ball_x_pos = ball_x_pos + ball_x_direction * ball_x_speed
	ball_y_pos = ball_y_pos + ball_y_direction * ball_y_speed
	score = 0

	# Check for collision
	if (ball_x_pos <= paddle_buffer + paddle_width and ball_y_pos + ball_height >= paddle1_y_pos and ball_y_pos - ball_height <= paddle1_y_pos + paddle_height):
		ball_x_direction = 1
	elif (ball_x_pos <= 0):
		ball_x_direction = 1 # -1 ?
		score = -1 
		return [score, paddle1_y_pos, paddle2_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]
	if (ball_x_pos >= window_width - paddle_width - paddle_buffer and ball_y_pos + ball_height >= paddle2_y_pos and ball_y_pos - ball_height <= paddle2_y_pos + paddle_height):
		ball_x_direction = -1
	elif(ball_x_pos >= window_width - ball_width):
		ball_x_direction = -1
		score = 1
		return [score, paddle1_y_pos, paddle2_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]
	if(ball_y_pos <= 0):
		ball_y_pos = 0
		ball_y_direction = 1
	elif(ball_y_pos >= window_height - ball_height):
		ball_y_pos = window_height - ball_height
		ball_y_direction = -1
		return [score, paddle1_y_pos, paddle2_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]

def update_paddle1(action, paddle1_y_pos):
	# if move up
	if(action[1] == 1):
		paddle1_y_pos = paddle1_y_pos - paddle_speed
	# if move down
	if(action[2] == 1):
		paddle1_y_pos = paddle1_y_pos + paddle_speed
	# dont let it move off the screen
	if(paddle1_y_pos < 0):
		paddle1_y_pos = 0
	if(paddle1_y_pos > window_height - paddle_height):
		paddle1_y_pos = window_height - paddle_height
	return paddle1_y_pos

def update_paddle2(action, ball_y_pos):
	# if move up
	if(action[1] == 1):
		paddle2_y_pos = paddle2_y_pos - paddle_speed
	# if move down
	if(action[2] == 1):
		paddle1_y_pos = paddle1_y_pos + paddle_speed
	# dont let it move off the screen
	if(paddle1_y_pos < 0):
		paddle1_y_pos = 0
	if(paddle1_y_pos > window_height - paddle_height):
		paddle1_y_pos = window_height - paddle_height
	return paddle1_y_pos

class Pong:
	def __init__(self):
		# random number for initial direction of ball
		num = random.randInt(0, 9)
		# keep score
		self.tally = 0
		# Initialize positions of our paddle
		self.paddle1_y_pos = window_height / 2 - paddle_height / 2
		self.paddle2_y_pos = window_height / 2 - paddle_height / 2
		# Ball direction definition
		self.ball_x_direction = 1
		self.ball_y_direction = 1
		# Starting point
		self.ball_x_pos = window_height / 2 - ball_width / 2

	def get_present_frame(self):
		pygame.event.pump()
		# make background black
		screen.fill(black)
		draw_paddle1(self.paddle1_y_pos)
		draw_paddle2(self.paddle2_y_pos)
		draw_ball(self.ball_x_pos, self.ball_y_pos)
		# get pixels
		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		# update the window
		pygame.display.flip()
		# return screen data
		return image_data

	def get_next_frame(self, action):
		pygame.event.pump()
		screen.fill(black)
		self.paddle1_y_pos = update_paddle1(action, self.paddle1_y_pos)
		draw_paddle1(self.paddle1_y_pos)
		self.paddle2_y_pos = update_paddle2(self.paddle2_y_pos, self.ball_y_pos)
		draw_ball(self.ball_x_pos, self.ball_y_pos)
		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		pygame.display.flip()
		self.tally = self.tally + score
		return [score, image_data]
		