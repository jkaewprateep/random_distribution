# random_distribution
sample random distribution you should know

1. It self learning from previous random result from the same node created.
2. Time vary is external variables for random function for time ago, they discoverd that random function should not including time as parameter it create some shared distribution value as result. 🧸💬 If we want random from single node or remote program we often use ```time % value``` that is because client connect with one source or the clock. 
3. 🐑💬 Hardware clock is accuracy but random function has more distribution ranges because it is accuracy from design function on vary current information and power input ( bus clock and response ), some technique measure running time of function response is blocked by time trying but can you vary time response with hardware⁉️ ( as reading about some security door and rubic )

```
import tensorflow as tf

from datetime import datetime

import matplotlib.pyplot as plt

dict_action = { "action" : [], "value" : [] }

def random_action_learning( ): 

	date = datetime.now()
	second = int(date.strftime("%S"))
	
	temp = tf.random.normal([1, 6], 0.2, 0.8, tf.float32).numpy()
	
	coeff_01 = ( second + 0 ) 
	coeff_02 = ( second + 10 ) 
	coeff_03 = ( second + 20 ) 
	coeff_04 = ( 40 - second ) 
	coeff_05 = ( 50 - second ) 
	coeff_06 = ( 60 - second ) 
	
	temp = tf.random.normal([1, 6], 1, 1, tf.float32)
	temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02, coeff_03, coeff_04, 
                                              coeff_05, coeff_06 ], shape=(6, 1), dtype=tf.float32))
	temp = tf.nn.softmax(temp[0])
	action = int(tf.math.argmax(temp))

	return action
```

![name-of-you-image](https://github.com/jkaewprateep/random_distribution/blob/main/Figure_1.png)



![name-of-you-image](https://github.com/jkaewprateep/random_distribution/blob/main/Figure_2.png)



![name-of-you-image](https://github.com/jkaewprateep/random_distribution/blob/main/FlappyBird_small.gif)



![name-of-you-image](https://github.com/jkaewprateep/random_distribution/blob/main/Pong%20Game.gif)
