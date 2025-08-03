import pygame as py
import random
import math
import time
import numpy as np
py.init()
def magnitude(x,y):
    return np.sqrt(x**2+y**2)
def normalize(x,y):
    m=magnitude(x,y)
    if m==0:
        return 0,0
    else:
        return x/m,y/m
def dot_product(x,y,x1,y1):
    return x*x1+y*y1
def clamp(val,min_val,max_val):
    return max(min(val,max_val),min_val)
#-------------------------------------------------------------------------------
class neural_network:
    def __init__(self,inp,h,o):#initialise network
        self.inputs=[1 for _ in range(inp)]
        lim=math.sqrt(7/(inp+o))#limit for weights
        self.hidden=[[[0,0] for _ in range(h[i])] for i in range(len(h))]#neuron value,bias
        self.outputs=[[0,0] for _ in range(o) ]
        self.weights=[]

        self.weights.append([[random.uniform(-lim,lim) for i in range(h[0])] for _ in range(inp)])

        for i in range(len(h)-1):
            self.weights.append([[random.uniform(-lim,lim) for n in range(h[i+1])] for w in range(h[i])])

        self.weights.append([[random.uniform(-lim,lim) for i in range(o)] for _ in range(h[-1])]) 
       
        #layer indexing is [layer][source neuron][target neuron]
   
    def softsign(self,n):#scaled softsign 
        return n/(1+abs(n))
    def scaled_softsign(self,n,s):
        return s*n/(1+abs(s*n))
    def scaled_tanh(self,n,s):
        return math.tanh(s*n)
    def leakyrelu(self,n):
        if n>0:
            return n
        else:
            return n*0.01
    def relu(self,n):
        return clamp(n,-1,1)

 
    def forward(self,inputvalues):#forward propagation- calculates outputs
        if len(inputvalues)!=len(self.inputs):
            raise ValueError("input size doesnt match input network size")
        self.inputs=inputvalues
        for h in range(len(self.hidden[0])):#iterating through neurons in hidden layer 0
            nsum=0#sum of all input neurons * weight connected to it
            for n in range(len(self.inputs)):#iterating through input neurons
                nsum+=self.inputs[n]*self.weights[0][n][h]
            nsum=self.softsign(nsum+self.hidden[0][h][1])#adds the bias then applies leaky relu activation
            self.hidden[0][h][0]=nsum
        if len(self.hidden)!=1:
            for layer in range(len(self.hidden)-1):
                for h in range(len(self.hidden[layer+1])):
                    nsum=0
                    for n in range(len(self.hidden[layer])):
                        nsum+=self.hidden[layer][n][0]*self.weights[layer+1][n][h]
                    nsum=self.softsign(nsum+self.hidden[layer+1][h][1])
                    self.hidden[layer+1][h][0]=nsum
        output_values=[0 for i in range(len(self.outputs))]
        for o in range(len(self.outputs)):#calculate output layer
            nsum=0
            for n in range(len(self.hidden[-1])):
                nsum+=self.hidden[-1][n][0]*self.weights[-1][n][o]
            nsum=self.scaled_tanh(nsum+self.outputs[o][1],2)
            self.outputs[o][0]=nsum
            output_values[o]=nsum
        return output_values
    
    def modifyby_evolution(self,mutation_rate):#random modifies weights and biases
        for w in range(len(self.weights)):
            for i in range(len(self.weights[w])):
                for j in range(len(self.weights[w][i])):
                    if random.uniform(0,1)<mutation_rate:
                        self.weights[w][i][j]+=random.uniform(-1,1)*mutation_rate # if mutation rate is higher it will also mutate by a higher amount
        for h in range(len(self.hidden)):
            for i in range(len(self.hidden[h])):
                if random.uniform(0,1)<mutation_rate:
                    self.hidden[h][i][1]+=random.uniform(-1,1)*mutation_rate
        for o in range(len(self.outputs)):
            if random.uniform(0,1)<mutation_rate:
                self.outputs[o][1]+=random.uniform(-1,1)*   mutation_rate
    def save_to_file(self,filename):
        with open(filename, 'w') as f:
            # Save network architecture
            f.write(f"{len(self.inputs)}\n")
            f.write(f"{len(self.hidden)}\n")  # Number of hidden layers
            for layer in self.hidden:
                f.write(f"{len(layer)}\n")
            f.write(f"{len(self.outputs)}\n")
            
            # Save neuron states and biases
            for layer in self.hidden:
                for neuron in layer:
                    f.write(f"{neuron[0]} {neuron[1]}\n")
            for neuron in self.outputs:
                f.write(f"{neuron[0]} {neuron[1]}\n")
                
            # Save weights
            for layer in self.weights:
                for row in layer:
                    f.write(" ".join(map(str, row)) + "\n")

    def load_from_file(self,filename):
        try:
            with open(filename, 'r') as f:
                # Load and validate architecture
                num_inputs = int(f.readline().strip())
                num_hidden_layers = int(f.readline().strip())
                hidden_sizes = [int(f.readline().strip()) for _ in range(num_hidden_layers)]
                num_outputs = int(f.readline().strip())
                
                # Verify network architecture matches
                if num_inputs != len(self.inputs) or \
                   num_hidden_layers != len(self.hidden) or \
                   num_outputs != len(self.outputs) or \
                   any(hidden_sizes[i] != len(self.hidden[i]) for i in range(num_hidden_layers)):
                    raise ValueError("Network architecture in file doesn't match current network")
                
                # Load neuron states and biases
                for layer in self.hidden:
                    for i in range(len(layer)):
                        val, bias = map(float, f.readline().strip().split())
                        layer[i] = [val, bias]
                
                for i in range(len(self.outputs)):
                    val, bias = map(float, f.readline().strip().split())
                    self.outputs[i] = [val, bias]
                
                # Load weights with validation
                for layer_idx in range(len(self.weights)):
                    source_size = len(self.weights[layer_idx])
                    target_size = len(self.weights[layer_idx][0])
                    for i in range(source_size):
                        weights = list(map(float, f.readline().strip().split()))
                        if len(weights) != target_size:
                            raise ValueError(f"Weight matrix size mismatch at layer {layer_idx}")
                        self.weights[layer_idx][i] = weights
                        
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error loading network: {str(e)}")
    def copy(self):
            new_net = neural_network(len(self.inputs), [len(h) for h in self.hidden], len(self.outputs))
            # Copy weights
            for l in range(len(self.weights)):
                for i in range(len(self.weights[l])):
                    for j in range(len(self.weights[l][i])):
                        new_net.weights[l][i][j] = self.weights[l][i][j]
            # Copy biases
            for l in range(len(self.hidden)):
                for i in range(len(self.hidden[l])):
                    new_net.hidden[l][i][1] = self.hidden[l][i][1]
            for i in range(len(self.outputs)):
                new_net.outputs[i][1] = self.outputs[i][1]
            return new_net
#-------------------------------------------------------------------------------------
class ball:    
    def __init__(self,x=500,y=250,mass=1,radius=15,colour=(255,255,255),speed=1):
        self.x=x
        self.y=y
        self.mass=mass
        self.radius=radius
        self.colour=colour
        self.speed=speed
        self.vx=0
        self.vy=0
        self.ax=0
        self.ay=0
        self.hit=False
        self.ePosColour=(255,150,150)#enemy possesion colour
        self.pPosColour=(150,150,255)
        self.colour=colour
        self.edgeColour=(self.colour[0]*0.75,self.colour[1]*0.75,self.colour[2]*0.75)
        self.friction=1
        self.damping=0.9 #enegry kept after bouncing
        self.epsilon=0.01
        self.isPlayers=False
    def draw(self):
        if self.isPlayers:
            self.colour=self.pPosColour
        else:
            self.colour=self.ePosColour
        self.edgeColour=(self.colour[0]*0.75,self.colour[1]*0.75,self.colour[2]*0.75)
        py.draw.circle(win, self.colour, (self.x, self.y), self.radius)
        py.draw.circle(win, self.edgeColour, (self.x, self.y), self.radius,5)
    def boundary_collision(self):
        xflag=False
        yflag=False
        if self.y<=self.radius:#top wall
            yflag=True
            self.y=self.radius+self.epsilon
            self.vy=-self.vy*self.damping
        if self.y>=winH-self.radius:#bottom wall
            yflag=True
            self.y=winH-self.radius-self.epsilon
            self.vy=-self.vy*self.damping
        if self.x<=self.radius:#left wall
            xflag=True
            self.x=self.radius+self.epsilon
            self.vx=-self.vx*self.damping
        if self.x>=winW-self.radius:#right wall
            xflag=True
            self.x=winW-self.radius-self.epsilon
            self.vx=-self.vx*self.damping
    def apply_force(self,fx,fy):
        self.vx+=fx/self.mass
        self.vy+=fy/self.mass
    def move(self,dt):
        friction_coeffiecient=-self.friction*self.mass*dt
        self.apply_force(self.vx*friction_coeffiecient,self.vy*friction_coeffiecient)
        self.vx+=self.ax*dt
        self.vy+=self.ay*dt
        self.x+=self.vx*dt*meters2pix
        self.y+=self.vy*dt*meters2pix
        self.boundary_collision()
   

class player:
    def __init__(self,x=500,y=250,mass=1,radius=30,colour=(89,100,217),speed=1,isPlayer=True,isAI=True):
        self.x=x
        self.y=y
        self.mass=mass
        self.radius=radius
        self.colour=colour
        self.speed=speed
        self.vx=0
        self.vy=0
        self.ax=0
        self.ay=0
        self.colour=colour
        self.edgeColour=(self.colour[0]*0.75,self.colour[1]*0.75,self.colour[2]*0.75)
        self.isPlayer=isPlayer
        self.isAI=isAI
        self.turn=False
        self.holding=False
        self.released=False
        #--------------AI parameters
        self.fitness=0
        self.score=0
        self.isElite=False
        self.gen=0
        #inp self x y opp x y ball x y goal y height out direction vector x,y
<<<<<<< HEAD
        self.net=neural_network(7,[4],2)
=======
        self.net=neural_network(4,[4],2)
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
        self.max_hit_reward=20
        self.hit_reward=2
        self.total_hit_reward=0
        self.game=None #training game this player is in
        self.last_vecx=0
        self.last_vecy=0
        self.repeat_reward=1
        self.similarity=0 #how similar the ai movement is to the desired movement
        self.predx,self.predy=0,0
        #--------------
        self.controlx=x
        self.controly=y
        self.friction=1
        self.damping=0.9 #enegry kept after bouncing
        self.epsilon=0.01
        self.max_proximity=150
        self.max_vel=self.speed/meters2pix*self.max_proximity
        
        self.count=0
        self.interval=10
    def draw(self):
        py.draw.circle(win, self.colour, (self.x, self.y), self.radius)
        py.draw.circle(win, self.edgeColour, (self.x, self.y), self.radius,5)
        if not self.isAI and self.turn:
           py.draw.circle(win, (200,200,200), (self.controlx, self.controly), 10,5)
           if self.holding:
              dx=self.x-self.controlx
              dy=self.y-self.controly
              mag=min(magnitude(dx,dy)*255/self.max_proximity,255)
              py.draw.line(win,(255,255-mag,100),(self.x,self.y),(self.controlx,self.controly),5)
        if self.isAI:  # Changed indentation and position
            vecx,vecy=self.x+self.predx*self.radius,self.y+self.predy*self.radius
            color = (
                    int(255 * (1 - self.similarity) / 2),  # red decreases with similarity
                    int(255 * (1 + self.similarity) / 2),  # green increases with similarity
                    100  # optional fixed blue
            )
            py.draw.line(win,color,(self.x,self.y),(vecx,vecy),5)
            label = font.render("AI", 1, (200,200,200))
            label_rect = label.get_rect(center=(self.x, self.y-self.radius/2))  # Center the text
            win.blit(label, label_rect)
            label = smallfont.render(f"FIT:{self.fitness:.2f}", 1, (200,200,200))
            label_rect = label.get_rect(center=(self.x, self.y))  # Center the text
            win.blit(label, label_rect)
            label = smallfont.render(f"GEN:{self.gen}", 1, (200,200,200))
            label_rect = label.get_rect(center=(self.x, self.y+self.radius/4))  # Center the text
            win.blit(label, label_rect)
            label = smallfont.render(f"REP:{self.repeat_reward}", 1, (200,200,200))
            label_rect = label.get_rect(center=(self.x, self.y+self.radius/2))  # Center the text
            win.blit(label, label_rect)
            label = smallfont.render(f"ACC:{self.similarity:.2f}", 1, (200,200,200))
            label_rect = label.get_rect(center=(self.x, self.y+self.radius/1.5))  # Center the text
            win.blit(label, label_rect)
            #draw arrow from centre to predicted position
           

            
    def save_network(self):
        file_name=f"{'player' if self.isPlayer else 'enemy'}_net_gen_{self.gen}.txt"
        self.net.save_to_file(file_name)
    def load_network(self,file_name):
        self.net.load_from_file(file_name)

    def boundary_collision(self):
        xflag=False
        yflag=False
        if self.y<=self.radius:#top wall
            yflag=True
            self.y=self.radius+self.epsilon
            self.vy=-self.vy*self.damping
        if self.y>=winH-self.radius:#bottom wall
            yflag=True
            self.y=winH-self.radius-self.epsilon
            self.vy=-self.vy*self.damping
        if self.x<=self.radius:#left wall
            xflag=True
            self.x=self.radius+self.epsilon
            self.vx=-self.vx*self.damping
        if self.x>=winW-self.radius:#right wall
            xflag=True
            self.x=winW-self.radius-self.epsilon
            self.vx=-self.vx*self.damping
    def circle_collision(self,circle):#circle is player or ball, both have same properties
        collision_flag=False
        dx=self.x-circle.x
        dy=self.y-circle.y
        dist=dx**2+dy**2
        collision_dist=(self.radius+circle.radius)**2
        
        if dist<=collision_dist:#COLLISION DETECTION
            collision_flag=True
            if type(circle)==ball:
                if self.total_hit_reward<self.max_hit_reward and self.turn==True:
                    self.score+=self.hit_reward
                    self.total_hit_reward+=self.hit_reward
                else:
                    pass#print('max hit reward reached')
                if self.turn==True:
                   circle.hit=True
                if self.isPlayer:
                    circle.isPlayers=True
                else:
                    circle.isPlayers=False
            nx,ny=normalize(dx,dy)
            correct_mag=self.radius+circle.radius
            #POSITION CORRECTION
            collision_x1=self.x-nx*self.radius
            collision_y1=self.y-ny*self.radius
            collision_x2=circle.x+nx*circle.radius
            collision_y2=circle.y+ny*circle.radius
            overlap=magnitude(collision_x1-collision_x2,collision_y1-collision_y2)
            adjustedx1=self.x+nx*overlap/1.9 #should be two but this is to prevent error
            adjustedy1=self.y+ny*overlap/1.9
            adjustedx2=circle.x-nx*overlap/1.9
            adjustedy2=circle.y-ny*overlap/1.9
            self.x=adjustedx1
            self.y=adjustedy1
            circle.x=adjustedx2
            circle.y=adjustedy2
            #------------------------------------------------------------------
            relative_vx=self.vx-circle.vx
            relative_vy=self.vy-circle.vy
            dotprod=dot_product(relative_vx,relative_vy,nx,ny)
            if dotprod>0:
                pass
            else:
                e=min(self.damping,circle.damping)
                j = -(1 + e) * dotprod / (1/self.mass + 1/circle.mass)
                impulse_x=j*nx
                impulse_y=j*ny
                self.apply_force(impulse_x,impulse_y)
                circle.apply_force(-impulse_x,-impulse_y)

        else:
            collision_flag=False
    def compute_fitness(self):
        k=5
        self.fitness=self.score/(k+self.score)

    def apply_force(self,fx,fy):
        self.vx+=fx/self.mass
        self.vy+=fy/self.mass
    def move(self,dt,ball,opp=None,goal=None):
        #----------------------------------------------------------movement
        if not self.isAI:#non ai movement
            mouse_buttons = py.mouse.get_pressed()
            temp_holding=self.holding
            # Print mouse info when left clicked
            if mouse_buttons[0]:  # Left mouse button
                #print(f"Mouse clicked at: {mouse_pos}")
                if self.turn:
                    self.holding=True
                    mouse_pos = py.mouse.get_pos()
                    self.controlx,self.controly=mouse_pos[0],mouse_pos[1]
            else:
                self.holding=False
                if temp_holding!=self.holding and self.released!=True:
                    #print('release')
                    self.released=True
                    self.max_vel=self.speed/meters2pix*self.max_proximity
                    self.vx+=clamp((self.x-self.controlx)/meters2pix *self.speed,-self.max_vel,self.max_vel)
                    self.vy+=clamp((self.y-self.controly)/meters2pix *self.speed,-self.max_vel,self.max_vel)
                self.controlx,self.controly=self.x,self.y
        #-------------------------------------------------------------------
        else:
            self.count+=1
            
            if self.turn==True and self.released==False:
                predx,predy=self.AImovement(opp,ball,goal) 
                tolerance=0.02
                predx,predy=normalize(predx,predy)
                self.predx,self.predy=predx,predy
                if abs(predx-self.last_vecx)<tolerance and abs(predy-self.last_vecy)<tolerance:
                    
                    #print('ai movement too similar to last movement')
                    self.score*=0.8
                self.last_vecx=predx
                self.last_vecy=predy
                self.vx+=predx
                self.vy+=predy
                #reward based on similarity to direction vector from player to ball
                player2ball_vecx=ball.x-self.x
                player2ball_vecy=ball.y-self.y
                player2ball_vecx,player2ball_vecy=normalize(player2ball_vecx,player2ball_vecy)
                dotprod=dot_product(predx,predy,player2ball_vecx,player2ball_vecy)
                self.similarity=dotprod
                reward = 0
                max_repeat = 10  # cap to prevent runaway growth

                if dotprod > 0.99:
                    self.repeat_reward += 1.0
                    reward = 5 * self.repeat_reward
                elif dotprod > 0.97:
                    self.repeat_reward += 0.5
                    reward = 3 * self.repeat_reward
                elif dotprod > 0.95:
                    self.repeat_reward += 0.2
                    reward = 1 * self.repeat_reward
                elif dotprod > 0.9:
                    # Slight reward, no repeat increase
                    reward = 0.5 * self.repeat_reward
                    # Optionally, decay repeat_reward slowly here:
                    self.repeat_reward = max(self.repeat_reward - 0.1, 0)
                elif dotprod > 0.5:
                    # Poor alignment — reset repeat_reward and penalize moderately
                    self.repeat_reward = 0
                    self.score /= 2
                else:
                    # Very poor alignment — reset and mutate with stronger penalty
                    self.repeat_reward = 0
                    self.score /= 10
                    self.mutate()

                # Cap repeat_reward so it doesn’t grow endlessly
                self.repeat_reward = min(self.repeat_reward, max_repeat)
                self.score+=reward
                self.released=True
            elif self.count% self.interval==0:
                self.predx,self.predy=self.AImovement(opp,ball,goal)
                self.predx,self.predy=normalize(self.predx,self.predy)
                player2ball_vecx=ball.x-self.x
                player2ball_vecy=ball.y-self.y
                player2ball_vecx,player2ball_vecy=normalize(player2ball_vecx,player2ball_vecy)
                dotprod=dot_product(self.predx,self.predy,player2ball_vecx,player2ball_vecy)
                self.similarity=dotprod
<<<<<<< HEAD
                if self.similarity<0.5 and self.gen>25:
                    self.score/=4
                    self.mutate()
                score=self.similarity-min((0.8+self.gen/50),0.999) #increase difficulty with generation
                if score>0.0001 and self.gen>50:
                    score*=1000
=======
                score=self.similarity-min((0.8+self.gen/500),0.9999) #increase difficulty with generation
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
                self.score+=score*dt*self.interval*5
                self.score=clamp(self.score,0,1000)
            
        friction_coeffiecient=-self.friction*self.mass*dt
        self.apply_force(self.vx*friction_coeffiecient,self.vy*friction_coeffiecient)
        self.vx+=self.ax*dt
        self.vy+=self.ay*dt
        self.x+=self.vx*dt*meters2pix
        self.y+=self.vy*dt*meters2pix
        self.boundary_collision()
        self.circle_collision(ball)
    def AImovement(self,opp,ball1,goal1):
        #feed inputs get 2 outputs
        self.win_mag=magnitude(winW,winH)
<<<<<<< HEAD
        out=self.net.forward([self.x/self.win_mag, self.y/self.win_mag, ball1.x/self.win_mag, ball1.y/self.win_mag, opp.x/self.win_mag, opp.y/self.win_mag, goal1.x/self.win_mag])
=======
        out=self.net.forward([self.x/self.win_mag,self.y/self.win_mag,ball1.x/self.win_mag,ball1.y/self.win_mag])
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
        vx=out[0]*self.max_vel
        vy=out[1]*self.max_vel
        #print(vx,vy)
        return vx,vy
         
    def mutate(self):
        rnd=random.uniform(0,1)
        if rnd<0.7 :    
            self.mutationrate=(1.01-self.fitness)/4
            self.net.modifyby_evolution(self.mutationrate)
            #print("mutation happened",self.mutationrate)
        elif rnd<0.8:
            #inp self x y opp x y ball x y goal y height out direction vector x,y
<<<<<<< HEAD
            self.net=neural_network(7,[4],2)
=======
            self.net=neural_network(4,[4],2)
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
            #print('reset network')
        elif rnd<0.9:
            self.game.copy_elite_genes(self)
            self.mutationrate=(1.01-self.fitness)/4
            self.net.modifyby_evolution(self.mutationrate)
            #print('elite genes copied')
        else:
            self.game.genetic_crossover(self)
            self.mutationrate=(1.01-self.fitness)/4
            self.net.modifyby_evolution(self.mutationrate)
            #print('elite genes crossed over with player')

class goal:
    def __init__(self,x=400,y=200,width=20,height=100,colour=(100,100,100),isPlayers=True):#if isplayers its the players side
        self.x=x
        self.y=y
        self.width=width
        self.height=height
        self.isPlayers=isPlayers
        self.goal_count=0
        self.colour=colour
    def draw(self):
        py.draw.rect(win, self.colour, (self.x, self.y,self.width,self.height))
    def rect_circle_collision(self,circle):
        closest_x=clamp(circle.x,self.x,self.x+self.width)
        closest_y=clamp(circle.y,self.y,self.y+self.height)

        dx=circle.x-closest_x
        dy=circle.y-closest_y

        if dx**2+dy**2<=circle.radius**2:
            if circle.isPlayers!=self.isPlayers:
                #print('player scored')
                self.goal_count+=1
            else:
                #print('own goal')
                self.goal_count+=1
                
class game:
<<<<<<< HEAD
    def __init__(self,playerRadius=70,ballRadius=50,goalHeight=0,ptw=3):
=======
    def __init__(self,playerRadius=70,ballRadius=50,goalHeight=150,ptw=3):
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
        self.pdefposx=winW/4
        self.edefposx=winW/4*3
        self.defposx=winW/2
        self.defposy=winH/2

        self.player1=player(self.pdefposx,self.defposy,20,playerRadius,(89,100,217),10,True,True)
        self.enemy1=player(self.edefposx,self.defposy,20,playerRadius,(217,58,27),10,False,True)
<<<<<<< HEAD
        self.random_posx=500
        self.random_posy=300
        self.ball1=ball(self.defposx+random.uniform(-self.random_posx,self.random_posx),self.defposy+random.uniform(-self.random_posy,self.random_posy),1,ballRadius,(230,230,230),1)
        #goalWidth=ballRadius
        goalWidth=0
=======
        self.random_posx=100
        self.random_posy=50
        self.ball1=ball(self.defposx+random.uniform(-self.random_posx,self.random_posx),self.defposy+random.uniform(-self.random_posy,self.random_posy),1,ballRadius,(230,230,230),1)
        goalWidth=ballRadius
        #goalWidth=0
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
        self.goal1=goal(0,winH/2-goalHeight/2,goalWidth,goalHeight,(100,100,220),True)
        self.goal2=goal(winW-goalWidth,winH/2-goalHeight/2,goalWidth,goalHeight,(220,100,100),False)
        self.playerGoals=self.goal2.goal_count
        self.enemyGoals=self.goal1.goal_count
        self.ptw=ptw
        #time constraints------------------------------------
        self.time1=time.time()
        self.time2=time.time()
        self.original_time=time.time()
        self.reset_interval=10
        self.game_time=100
        self.plrecordx,self.plrecordy=self.player1.x,self.player1.y
        self.elrecordx,self.elrecordy=self.enemy1.x,self.enemy1.y
        self.blrecordx,self.blrecordy=self.ball1.x,self.ball1.y
        #------------------------------------------------------
        self.avgFit=0
        self.vel_threshold=0.05
        self.turn=round(random.random())
        #print('t',self.turn)
        self.player1.turn=True if self.turn==0 else False
        self.enemy1.turn=True if self.turn==1 else False
        self.ball1.isPlayers=True if self.turn==0 else False
        self.visible=True
        self.game=None #training game this game is in
    def record_locations(self):
        self.plrecordx,self.plrecordy=self.player1.x,self.player1.y
        self.elrecordx,self.elrecordy=self.enemy1.x,self.enemy1.y
        self.blrecordx,self.blrecordy=self.ball1.x,self.ball1.y
    def draw(self):
        self.player1.draw()
        self.enemy1.draw()
        self.ball1.draw()
        self.goal1.draw()
        self.goal2.draw()
    def move(self,dt):
        if self.game is not None:
             self.player1.game=self.game
             self.enemy1.game=self.game
        self.time2=time.time()
        self.ball1.move(dt)
        self.player1.move(dt,self.ball1,self.enemy1,self.goal2)
        #self.player1.circle_collision(self.enemy1)
        self.enemy1.move(dt,self.ball1,self.player1,self.goal1)
        self.goal1.rect_circle_collision(self.ball1)
        self.goal2.rect_circle_collision(self.ball1)
        if (abs(self.ball1.vx+self.ball1.vy)<=self.vel_threshold):
            if (self.player1.released==True or self.enemy1.released==True) and (abs(self.player1.vx+self.player1.vy)<self.vel_threshold and abs(self.enemy1.vx+self.enemy1.vy)<self.vel_threshold):
                #print('ball stopped moving')
                self.ball1.hit=False
                self.swap_turns()
       
      
        bpg=self.playerGoals
        beg=self.enemyGoals
        self.playerGoals=self.goal2.goal_count
        self.enemyGoals=self.goal1.goal_count
        if bpg !=self.playerGoals or beg!=self.enemyGoals:
            #print('goal change')
            if bpg!=self.playerGoals:
<<<<<<< HEAD
                self.player1.score*=2
                self.enemy1.score/=2
            else:
                self.enemy1.score*=2
                self.player1.score/=2
=======
                self.player1.score+=1
                self.enemy1.score/=1.2
            else:
                self.enemy1.score+=1
                self.player1.score/=1.2
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
            
            self.partial_reset()
            #print('player goals:',self.playerGoals,self.enemyGoals)
            if self.playerGoals>=self.ptw:
                self.full_reset()
                #print('player won')
            elif self.enemyGoals>=self.ptw:
                self.full_reset()
                #print('enemy won')
        else:
            if (self.player1.isAI and self.enemy1.isAI):
                if self.time2-self.time1>=self.reset_interval:
                    if time.time()-self.original_time<=self.reset_interval*2+0.1:
                        if (self.ball1.x-self.defposx)**2+(self.ball1.y-self.defposy)**2<10:
                            self.partial_reset()
                            self.full_reset()
                            #print('reset due to lack of ball movement')
                        elif (self.player1.x-self.pdefposx)**2+(self.player1.y-self.defposy)**2<20:
                            self.partial_reset()
                            self.full_reset()
                            #print('reset due to lack of player movement')
                        elif (self.enemy1.x-self.edefposx)**2+(self.enemy1.y-self.defposy)**2<20:
                            self.partial_reset()
                            self.full_reset()
                            #print('reset due to lack of enemy movement')
                    self.tplrecordx,self.tplrecordy=self.plrecordx,self.plrecordy
                    self.telrecordx,self.telrecordy=self.elrecordx,self.elrecordy
                    self.tblrecordx,self.tblrecordy=self.blrecordx,self.blrecordy
                    self.record_locations()
                    pdx,pdy=self.player1.x-self.tplrecordx,self.plrecordy-self.tplrecordy
                    edx,edy=self.elrecordx-self.telrecordx,self.elrecordy-self.telrecordy
                    bdx,bdy=self.blrecordx-self.tblrecordx,self.blrecordy-self.tblrecordy
                    pmag=pdx**2+pdy**2
                    emag=edx**2+edy**2
                    bmag=bdx**2+bdy**2
                    #print(pmag,emag,bmag)
                    if (pmag<10 or emag<10 or bmag<10) and time.time()-self.original_time>=self.reset_interval*2:
                        self.partial_reset()
                        self.full_reset()
                        #print('reset due to lack of movement')
                    self.time1=time.time()
                    self.time2=time.time()
                

        
        self.current_time=time.time()
        if self.current_time-self.original_time>=self.game_time:
            self.partial_reset()
            self.full_reset()
            #print('reset as game has finished')

        self.player1.score+=dt/300
        self.enemy1.score+=dt/300
        self.reward_goal_proximity(dt)
        self.player1.compute_fitness()
        self.enemy1.compute_fitness()
        self.avg_fitness()
        if self.visible:
            self.draw()

    def save_networks(self):
<<<<<<< HEAD
        if self.player1.isAI and self.player1.isElite :
=======
        if self.player1.isAI and self.player1.isElite:
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
            self.player1.save_network()
        if self.enemy1.isAI and self.enemy1.isElite:
            self.enemy1.save_network()  
    def partial_reset(self):#when goal is scored
        self.ball1.ax,self.ball1.ay,self.ball1.vx,self.ball1.vy=0,0,0,0
        self.player1.ax,self.player1.ay,self.player1.vx,self.player1.vy=0,0,0,0
        self.enemy1.ax,self.enemy1.ay,self.enemy1.vx,self.enemy1.vy=0,0,0,0
        self.ball1.x,self.ball1.y=self.defposx+random.uniform(-self.random_posx,self.random_posx),self.defposy+random.uniform(-self.random_posy,self.random_posy)
        self.player1.x,self.player1.y=self.pdefposx,self.defposy
        self.enemy1.x,self.enemy1.y=self.edefposx,self.defposy
        self.turn=round(random.random())
        #print('t',self.turn)
        self.player1.turn=True if self.turn==0 else False
        self.enemy1.turn=True if self.turn==1 else False
        self.ball1.isPlayers=True if self.turn==0 else False
        self.ball1.hit=False
        self.player1.repeat_reward=0
        self.enemy1.repeat_reward=0
        #print('partial reset')
    def full_reset(self):
        self.playerGoals=0
        self.enemyGoals=0
        self.goal1.goal_count=0
        self.goal2.goal_count=0
        self.player1.fitness=0
        self.player1.score=0
        self.enemy1.fitness=0
        self.enemy1.score=0
        self.time1=time.time()
        self.time2=time.time()
        self.original_time=time.time()
        if not self.player1.isElite:
           self.player1.mutate()
        if not self.enemy1.isElite:
           self.enemy1.mutate()
        self.player1.gen+=1
        self.enemy1.gen+=1
        
        self.save_networks()
        #print('full reset')
    def avg_fitness(self):
        self.avgFit=(self.player1.fitness+self.enemy1.fitness)/2

    def swap_turns(self):
        self.turn=0 if self.turn==1 else 1
        self.player1.turn=True if self.player1.turn==False else False
        self.enemy1.turn=True if self.enemy1.turn==False else False
        self.ball1.isPlayers=True if self.player1.turn==True else False
        self.player1.released=False 
        self.enemy1.released=False
    def txtstr(self):
        return f" | Player goals {self.playerGoals} : {self.enemyGoals} | Turn: {'players' if self.turn==0 else 'enemys'} | points to win {self.ptw} | current game time: {time.time()-self.original_time:.1f} out of {self.game_time}|"     
    def reward_goal_proximity(self,dt):
        g1distance = magnitude(self.ball1.x-self.goal1.x, self.ball1.y-self.goal1.y) 
        g2distance = magnitude(self.ball1.x-self.goal2.x, self.ball1.y-self.goal2.y)
        self.window_mag = magnitude(winW, winH)
        
        # Normalize distances to 0-1 range and invert so closer = higher reward
        g1distance = 1 - (g1distance/self.window_mag)  # For player scoring in goal1 
        g2distance = 1 - (g2distance/self.window_mag)  # For enemy scoring in goal2
        
        reward_coefficient = dt
        min_proximity=400
        
        # Player is rewarded for ball being close to goal1 
        if g1distance < min_proximity/self.window_mag:
            self.player1.score += g1distance * reward_coefficient
            if self.visible:
                pass#print("Player reward:", g1distance * reward_coefficient)
                
        # Enemy is rewarded for ball being close to goal2
        if g2distance < min_proximity/self.window_mag:
            self.enemy1.score += g2distance * reward_coefficient
        
class training_game:
    def __init__(self,number=10):
        self.games=[game() for _ in range(number)]
<<<<<<< HEAD
        self.games[0].player1.net.load_from_file("enemy_net_gen_50.txt")#load player net, giving a head start to the first game
        self.games[0].enemy1.net.load_from_file("enemy_net_gen_50.txt")#load player net
        self.games[1].player1.net.load_from_file("enemy_net_gen_25.txt")#load player net, giving a head start to the first game
        self.games[1].enemy1.net.load_from_file("enemy_net_gen_25.txt")#load player net
=======
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
        self.eliteIndexes=[None]#indexes of elite games
        self.elitesNumber=round(number/10)
        self.find_elites(self.elitesNumber)
        self.gameSpeed=1
    def run(self,dt):
        dt*=self.gameSpeed
        self.find_elites(self.elitesNumber)
        for i,g in enumerate(self.games):
            g.game=self
            g.move(dt)
            if i ==self.eliteIndexes[0]:
                g.visible=True
            else:
                g.visible=False

    def txtstr(self):
        txt=self.games[self.eliteIndexes[0]].txtstr()
        return txt+" game index:"+str(self.eliteIndexes[0])+f" | avg fit:{self.games[self.eliteIndexes[0]].avgFit:.2f}"
    def find_elites(self, n=2):
        # Sort indexes of games by avgFit descending
        sorted_indexes = sorted(range(len(self.games)), key=lambda i: self.games[i].avgFit, reverse=True)
        # Take top n elites
        self.eliteIndexes = sorted_indexes[:n]

        # Set isElite flags for all games
        for i, game in enumerate(self.games):
            is_elite = i in self.eliteIndexes
            game.player1.isElite = is_elite
            game.enemy1.isElite = is_elite

    def copy_elite_genes(self,player):
        rand=random.randint(0, len(self.eliteIndexes)-1)
        elite_game=self.games[self.eliteIndexes[rand]]
        if player.isPlayer:
            player.net=elite_game.player1.net.copy()
        else:
            player.net=elite_game.enemy1.net.copy()
    def genetic_crossover(self,player1):
        rand = random.randint(0, len(self.eliteIndexes)-1)
        elite_game = self.games[self.eliteIndexes[rand]]
        net1 = player1.net.copy()
        net2 = elite_game.player1.net.copy() if player1.isPlayer else elite_game.enemy1.net.copy()
                
        # Safely perform crossover between weights
        for layer in range(min(len(net1.weights), len(net2.weights))):
            for i in range(min(len(net1.weights[layer]), len(net2.weights[layer]))):
                for j in range(min(len(net1.weights[layer][i]), len(net2.weights[layer][i]))):
                    if random.random() < 0.5:
                        net1.weights[layer][i][j] = net2.weights[layer][i][j]
        
        # Safely perform crossover between hidden layer biases
        for layer in range(min(len(net1.hidden), len(net2.hidden))):
            for i in range(min(len(net1.hidden[layer]), len(net2.hidden[layer]))):
                if random.random() < 0.5:
                    net1.hidden[layer][i][1] = net2.hidden[layer][i][1]
        
        # Safely perform crossover between output layer biases
        for i in range(min(len(net1.outputs), len(net2.outputs))):
            if random.random() < 0.5:
                net1.outputs[i][1] = net2.outputs[i][1]
        
        player1.net = net1
        

        
       
        
            

#CONSTANTS
winW=1500
winH=750
meters2pix=3800
#--------------
win=py.display.set_mode((winW,winH))
py.display.set_caption("football")
<<<<<<< HEAD
font=py.font.Font("SpaceMono-Regular.ttf", 18)
smallfont=py.font.Font("SpaceMono-Regular.ttf", 12)
=======
font=py.font.SysFont(None, 18)
smallfont=py.font.SysFont(None, 15)
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
game1=training_game(200)
game1.elitesNumber=20
game2=game()
game2.player1.isAI=False
game2.enemy1.isAI=True
<<<<<<< HEAD
game2.enemy1.net.load_from_file("player_net_gen_31.txt")#load player net
=======
game2.enemy1.net.load_from_file("player_net_gen_155.txt")#load player net
>>>>>>> d003124207b26435693a5ee9ad4c6edb824b19df
fps=100
dt=1/fps
txtstr=f"FPS:{fps}"
txt=font.render(txtstr,1,(255,255,255),(92*0.75,217*0.75,107*0.75))
ltime=time.time()-dt#current time
ctime=time.time()
count=0
interval=50
run=True
while run:
    ltime=time.time()
    py.time.delay(int(1000/fps))
    win.fill((92,217,107))
    
    
    for event in py.event.get():  
        if event.type == py.QUIT:
            run = False  # Ends the game loop
    if count%interval==0:
       txtstr=f"FPS:{1/dt:.1f}"
    game2.move(dt)
    txt=font.render(txtstr+"  "+game1.txtstr(),1,(255,255,255),(92*0.75,217*0.75,107*0.75))
    #game1.run(dt)
    #game1.move(dt)
    win.blit(txt,(0,0))
    ctime=time.time()
    dt=ctime-ltime
    count+=1
    py.display.update()

py.quit()
