import numpy as np

##------------------------Useful Functions------------------------------------------------##
    
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    '''
    input images, shape = (N,C,H,W)
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    output col, shape = (N*out_h*out_w,C*filter_h*filter_w)
    '''
    # Padding
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1  # 输出数据的高
    out_w = (W + 2*pad - filter_w)//stride + 1  # 输出数据的长
    
    img = np.pad(input_data,((0,0),(0,0),(pad,pad),(pad,pad)),"constant") # 两侧填充
    # shape = (N,C,H+2*pad,W+2*pad)
    col = np.empty((filter_h,filter_w,N,C,out_h,out_w))
    for y in range(filter_h):
        for x in range(filter_w):
            col[y, x] = img[:,:,y:out_h*stride+y:stride,x:out_w*stride+x:stride]
    col = col.transpose(2, 4, 5, 3, 0, 1).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    '''
    inverse operation to im2col
    input col, shape = (N*out_h*out_w,C*filter_h*filter_w)
    output images, shape = (N,C,H,W)
    '''
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # shape = (N*out_h*out_w, C*filter_h*filter_w)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(4, 5, 0, 3, 1, 2)
    # now: shape = (filter_h, filter_w, N, C, out_h, out_w)
    
    img = np.zeros((N, C, H+2*pad, W+2*pad))
    for y in range(filter_h):
        for x in range(filter_w):
            img[:,:,y:out_h*stride+y:stride,x:out_w*stride+x:stride] += col[y, x]

    return img[:, :, pad:H+pad, pad:W+pad]


##------------------------Useful Functions----------------------------------------------##
























##------------------------Optimizers-----------------------------------------------------##




class SGD:
    '''
    SGD optimizer with momentum
    update equation:
    d <- beta*d+(1-beta)*g
    w <= w - lr*d
    '''
    def __init__(self,LEARNING_RATE=0.001,MOMENTUM=0.0):
        self.modules = []
        self.direct = []
        self.lr = LEARNING_RATE
        self.beta = MOMENTUM
        
    def add_module(self, module):
        self.modules.append(module)
        self.direct.append(dict())
        for key in module.state_dict:
            param = module.state_dict[key]['value']
            self.direct[-1][key] = np.zeros_like(param)
    
    def step(self):
        for i in range(len(self.modules)):
            for key in self.modules[i].state_dict:
                g = self.modules[i].state_dict[key]['grad']
                self.direct[i][key] *= self.beta
                self.direct[i][key] += (1-self.beta)*g
                self.modules[i].state_dict[key]['value'] -= self.lr*self.direct[i][key]
        return 'success'

     
class Adam:
    '''
    Adam optimizer
    update equation:
    s <- beta1*d+(1-beta1)*g
    r <- beta2*(g**2)+(1-beta)*(g**2)
    d <- s/r
    w <- w - lr*d
    '''
    def __init__(self,LEARNING_RATE=0.01,BETA1 = 0.9,BETA2 = 0.999):
        self.modules = []
        self.direct = []
        self.t = 0
        self.lr = LEARNING_RATE
        self.beta1 = BETA1
        self.beta2 = BETA2
        
    def add_module(self, module):
        self.modules.append(module)
        self.direct.append(dict())
        for key in module.state_dict:
            param = module.state_dict[key]['value']
            self.direct[-1][key] = {'s':np.zeros_like(param),
                                    'r':np.zeros_like(param)}
    
    def step(self):
        self.t += 1
        for i in range(len(self.modules)):
            for key in self.modules[i].state_dict:
                g = self.modules[i].state_dict[key]['grad']
                self.direct[i][key]['s'] *= self.beta1
                self.direct[i][key]['r'] *= self.beta2
                self.direct[i][key]['s'] += (1-self.beta1)*g
                self.direct[i][key]['r'] += (1-self.beta2)*(g**2)
                
                s_hat = self.direct[i][key]['s']/(1-self.beta1**self.t)
                r_hat = self.direct[i][key]['r']/(1-self.beta2**self.t)
                
                d = s_hat/(np.sqrt(r_hat)+1e-8)
                self.modules[i].state_dict[key]['value'] -= self.lr*d
        return 'success'
        

##------------------------Optimizers--------------------------------------------------##



































##------------------------Neural Networks-----------------------------------------##


class Module:
    '''
    这个是基础模块，所有神经网络模块将从这里衍生
    包含了基础的state_dict字典，它将包含模块中所有可
    训练的参数和参数的累计梯度
    forward和backward将用于网络的前向计算和反向传播
    call将直接调用forward
    zero_grad消除累计梯度，在迭代计算中需要经常调用
    '''
    def __init__(self):
        self.state_dict = {}
    
    def forward(self, x):
        pass
    
    def backward(self, dz):
        pass
        
    def zero_grad(self):
        pass
    
    def __call__(self, x):
        return self.forward(x)

class Linear(Module):
    '''
    Linear层的参数是trainable的，因此你不但要实现正向和反向
    传播算法，还要实现一个基于梯度更新参数的过程。
    这个过程在Pytorch中，梯度是tensor的一个属性，额外的optimizer
    基于梯度对tensor进行修正，不断迭代进行优化
    我们这里实现公认非常高效的Adam优化器
    参数的初始化使用xavier初始化
    '''
    def __init__(self,
                 in_features,
                 out_features):
        '''
        in_features: size of each input sample
 |      out_features: size of each output sample
        '''
        super().__init__()
        
        self._in_features = in_features
        self._out_features = out_features
        
        a = np.sqrt(5).item()
        bound = np.sqrt(6/((1+a**2)*in_features)).item()
        
        self.weight = (np.random.rand(in_features,out_features)-0.5)*2*bound
        self.bias  = np.zeros(out_features)
        
        self._dw = np.zeros_like(self.weight)
        self._db = np.zeros_like(self.bias)
        
        self.state_dict['weight'] = {'value':self.weight,'grad':self._dw}
        self.state_dict['bias'] = {'value':self.bias,'grad':self._db}
        
        self._latest_input = None
        
    def forward(self, x):
        '''
        calculate matrix multiply, and add a bias
        '''
        self._latest_input = x
        out =  np.dot(x,self.weight)+self.bias
        return out
    
    def backward(self, dz):
        '''
        dz: gradient in backpropagation
        call backward you will get gradient of weight and bias
        they are saved in self.state_dict['weight']['grad'] and
        self.state_dict['bias']['grad']
        '''
        batch_size,_ = self._latest_input.shape
        
        g_of_w = np.dot(self._latest_input.T,dz)
        g_of_b = np.sum(dz, axis = 0)
        
        self._dw += g_of_w
        self._db += g_of_b
        
        dx = np.dot(dz,self.weight.T)
        
        return dx
        
    def zero_grad(self):
        '''
        zeros the gradients of all parameters
        '''
        self._dw -= self._dw
        self._db -= self._db

        
class ReLU(Module):
    '''
    relu linear rectify function,f(x) = max(x,0)
    '''
    def __init__(self):
        super().__init__()
        self.mask = None
        
    def forward(self, x):
        '''
        set a mask
        '''
        self.mask = x < 0
        x[self.mask] = 0
        return x
    
    def backward(self, dz):
        '''
        zeros gradients of mask
        '''
        dz[self.mask] = 0
        return dz
 
 
class LeakyReLU(Module):
    '''
    relu linear rectify function with leak
    f(x) = x if x>0 else slope*x
    '''
    def __init__(self,slope):
        super().__init__()
        self.mask = None
        self.slope = slope
        
    def forward(self, x):
        self.mask = x < 0
        x[self.mask] *= self.slope
        return x
    
    def backward(self, dz):
        dz[self.mask] *= self.slope
        return dz
     

class Sigmoid(Module):
    '''
    sigmoid activate function
    return values in [0,1]
    '''
    def __init__(self):
        super().__init__()
        self.out = None
        
    def forward(self, x):
        '''
        f(x) = 1/(1+np.exp(-x))
        '''
        self.out = 1/(1+np.exp(-x))
        return self.out
    
    def backward(self, dz):
        '''
        f'(x) = f(x)*(1-f(x))
        '''
        return dz*self.out*(1-self.out)


class Tanh(Module):
    '''
    tanh activate function
    return values in [-1,1]
    '''
    def __init__(self):
        super().__init__()
        self.out = None
        
    def forward(self, x):
        '''
        f(x) = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        '''
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, dz):
        '''
        f'(x) = (1-f(x)*f(x))
        '''
        return dz*(1-self.out**2)
   
        
class Sequential(Module):
    '''
    Sequentially connect many modules
    '''
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        
    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
    
    def backward(self, dz):
        for module in self.modules[::-1]:
            dz = module.backward(dz)
        return dz
                
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
            
    def apply_optim(self, optimizer):
        for module in self.modules:
            optimizer.add_module(module)

    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3,
                 stride=1, padding=0):
        '''
        初始化权重（卷积核4维）、偏置、步幅、填充
        '''
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._pad = padding
        
        
        fan_in = in_channels*kernel_size*kernel_size
        
        a = np.sqrt(5).item()
        bound = np.sqrt(6/((1+a**2)*fan_in)).item()
        # 可学习参数
        self.weights = (np.random.rand(out_channels,in_channels,
                    kernel_size,kernel_size)-0.5)*2*bound
        
        self.bias = np.zeros(out_channels)
        
        self._dw = np.zeros_like(self.weights)
        self._db = np.zeros_like(self.bias)
        
        self.state_dict['weights'] = {'value':self.weights,'grad':self._dw}
        self.state_dict['bias'] = {'value':self.bias,'grad':self._db}
        
        # 中间数据（backward时使用）
        self._latest_input = None   
        self._col_x = None
        self._col_weights = None
        
    def forward(self, x):
        # 数据大小
        N, C, H, W = x.shape
        # 卷积核大小
        FN, C, FH, FW = self.weights.shape
        # 计算输出数据大小
        out_h = 1 + (H + 2*self._pad - FH) // self._stride
        out_w = 1 + (W + 2*self._pad - FW) // self._stride
        # 利用im2col将输入x转换为col矩阵
        self._col_x = im2col(x, FH, FW, self._stride, self._pad)
        # 卷积核转换为列，展开为2维数组
        self._col_weights = self.weights.reshape(FN, -1).T
        # 计算正向传播
        out = np.dot(self._col_x, self._col_weights) + self.bias
        # out.shape = (N*out_h*out_w,FN)
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2) 
        # out.shape = (N,FN,out_h,out_w)

        self._latest_input = x
        
        return out

    def backward(self, dz):
        # 卷积核大小
        FN, C, FH, FW = self.weights.shape
        # dz.shape = (N,FN,out_h,out_w)
        dz = dz.transpose(0,2,3,1).reshape(-1, FN)
        # size = (N*out_h*out_w,FN)

        g_of_b = np.sum(dz, axis=0)
        # size(FN,) 与bias的维度相同
        g_of_w = np.dot(self._col_x.T, dz)
        # size = mm(size(C*ker*ker,N*out_h*out_w), (N*out_h*out_w,FN))
        # = size(C*ker*ker, FN)
        g_of_w = g_of_w.transpose(1, 0).reshape(FN, C, FH, FW)
        # size(FN,C,ker,ker), 与weights的维度相同
        
        self._dw += g_of_w
        self._db += g_of_b

        g_of_col_x = np.dot(dz, self._col_weights.T)
        # size = mm(size(N*out_h*out_w,FN), (FN,C*ker*ker))
        # = size(N*out_h*out_w, C*ker*ker)
        # 经过col2im转换为图像
        dx = col2im(g_of_col_x, self._latest_input.shape, FH, FW, self._stride, self._pad)
        # size(N,C,H,W) = self.x.shape

        return dx
        
    def zero_grad(self):
        # 清零dW和db
        self._dw -= self._dw
        self._db -= self._db
        
class MaxPool2d(Module):
    def __init__(self, kernel_size = 2, stride = 2, padding=0):
        super().__init__()
        self._pool_h = kernel_size
        self._pool_w = kernel_size
        self._stride = stride
        self._pad = padding
        
        self._latest_input = None
        self._arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self._pool_h + 2*self._pad)//self._stride+1
        out_w = (W - self._pool_w + 2*self._pad)//self._stride+1
        # 用im2col展开图像为二维矩阵
        col_x = im2col(x, self._pool_h, self._pool_w, self._stride, self._pad) 
        # col_x.shape = (N*out_h*out_w,C*ker*ker)
        col_x = col_x.reshape(-1, self._pool_h*self._pool_w) 
        # col_x.shape = (N*out_h*out_w*C,ker*ker)
        # 在每行取最大值,计算argmax并保存下来
        arg_max = np.argmax(col_x, axis=1)
        out = col_x[range(col_x.shape[0]),arg_max]
        # col_x.shape = (N*out_h*out_w*C,)
        # 重塑成图像
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) 
        # size = (N,C,out_h,out_w)

        self._latest_input = x
        self._arg_max = arg_max

        return out

    def backward(self, dz):
        N,C,out_h,out_w = dz.shape
        dz = dz.transpose(0, 2, 3, 1) 
        # size = (N,out_h,out_w,C)
        pool_size = self._pool_h * self._pool_w
        n = np.prod(dz.shape)
        dmax = np.zeros((n, pool_size))
        # 把之前最大元素对应位置用dz填充, 其他置0
        dmax[np.arange(n), self._arg_max.flatten()] = dz.flatten()
        # size = (N*out_h*out_w*C,ker*ker)
        dcol = dmax.reshape(N * out_h * out_w, -1)
        # size = (N*out_h*out_w,C*ker*ker)
        # 调用col2im复原图像
        dx = col2im(dcol, self._latest_input.shape, self._pool_h,
                    self._pool_w, self._stride, self._pad)
        # size = (N,C,H,W)
        return dx
        
        
class Flatten(Module):
    '''
    展平图像，(N,C,H,W)->(N,C*H*W)
    '''
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.output_shape = None
        
    def forward(self, x):
        self.input_shape = x.shape
        out = x.reshape(x.shape[0],-1)
        self.output_shape = out.shape
        return out
    
    def backward(self, dz):
        assert self.output_shape == dz.shape
        dx = dz.reshape(self.input_shape)
        return dx
        
        
class RNN(Module):
    '''
    单隐层RNN, 使用两个线性层和一个tanh激活函数
    h_{t} = tanh(mm(x_{t},U)+mm(h_{t-1},W)+b)
    提供单输出和对最终时间输出h_t的反向传播
    '''
    def __init__(self, input_size, hidden_size, requires_clip = False):
        super().__init__()
        
        self._input_size = input_size
        self._hidden_size = hidden_size
        self.requires_clip = requires_clip
        
        self.weight_ih = np.random.randn(input_size,hidden_size)/(np.sqrt(input_size)/2)
        self.weight_hh = np.random.randn(hidden_size,hidden_size)/(np.sqrt(hidden_size)/2)
        self.bias = np.zeros(hidden_size)
        
        # 反向传播时可能用到的中间变量
        self._x_t = None # t时刻的输入x
        self._h_t = None # t时刻的隐层h
        self._h_t_no_tanh = None # t时刻的x.dot(weight_ih)+h.dot(weight_hh)+bias
        
        # 梯度量
        self._dw_ih = np.zeros_like(self.weight_ih)
        self._dw_hh = np.zeros_like(self.weight_hh)
        self._db = np.zeros_like(self.bias)
        
        self.state_dict['weight_ih'] = {'value':self.weight_ih,'grad':self._dw_ih}
        self.state_dict['weight_hh'] = {'value':self.weight_hh,'grad':self._dw_hh}
        self.state_dict['bias'] = {'value':self.bias,'grad':self._db}
        
    def forward(self, x, h_0 = None):
        '''
        x: np array, shape = (batch_size, seq_len, input_size)
        h_0: hidden unit, None for all zero
        out: h_t, shape = (batch_size, hidden_size)
        '''
        n,seq_len,m = x.shape
        assert m==self._input_size
        if h_0 is None:
            h_0 = np.zeros((n,self._hidden_size))
            
        self._x_t = x.copy()
        self._h_t = np.empty((n,seq_len+1,self._hidden_size))
        self._h_t_no_tanh = np.empty((n,seq_len,self._hidden_size))
        
        h_t = h_0
        for t in range(seq_len):
            x_t = x[:,t,:]
            self._h_t[:,t,:] = h_t
            self._h_t_no_tanh[:,t,:] = x_t.dot(self.weight_ih)+h_t.dot(self.weight_hh)+self.bias
            h_t = np.tanh(self._h_t_no_tanh[:,t,:])
        
        self._h_t[:,-1,:] = h_t
        self._seq_len = seq_len
        
        return h_t
    
    def backward(self, dh):
        '''
        dh: np array, shape = (batch_size, hidden_size)
        gradient of final hidden
        calculate bptt
        '''
        h_t = self._h_t[:,-1,:]
        for t in range(self._seq_len-1,-1,-1):
            do = dh*(1-h_t**2)
            
            h_t = self._h_t[:,t,:]
            x_t = self._x_t[:,t,:]
            
            dw_ih_t = x_t.T.dot(do)
            dw_hh_t = h_t.T.dot(do)
            db_t = np.sum(do,axis = 0)
            dh = do.dot(self.weight_hh.T)
            
            self._dw_ih += dw_ih_t
            self._dw_hh += dw_hh_t
            self._db += db_t
        
        if self.requires_clip:
            self.clip()
        
        return 'success'
    
    
    def clip(self, threshold = 2.):
        clipped_dw_ih = np.clip(self._dw_ih,-threshold,threshold)
        clipped_dw_hh = np.clip(self._dw_hh,-threshold,threshold)
        clipped_db = np.clip(self._db,-threshold,threshold)
        
        self.zero_grad()
        self._dw_ih += clipped_dw_ih
        self._dw_hh += clipped_dw_hh
        self._db += clipped_db
        
    def zero_grad(self):
        self._dw_ih -= self._dw_ih
        self._dw_hh -= self._dw_hh
        self._db -= self._db
        
        
class BatchNorm(Module):
    '''
    1D的batchnorm，含有可学习参数w和b来复原分布
    我们把计算步骤全部写出来方便我们计算反向传播
    x_mean = (1/n)*sum(x,axis=1)
    x_cent = x-x_mean
    x_square = x_cent**2
    x_var = (1/n)*sum(x_square)
    x_std = sqrt(x_var)
    x_hat = x_cent/x_std
    y = w*x_hat+b
    '''
    def __init__(self, num_features):
        super().__init__()
        
        self.weight = np.ones(num_features)
        self.bias = np.zeros(num_features)
        
        self._dw = np.zeros_like(self.weight)
        self._db = np.zeros_like(self.bias)
        
        self.state_dict['weight'] = {'value':self.weight,'grad':self._dw}
        self.state_dict['bias'] = {'value':self.bias,'grad':self._db}
        
        self.m = num_features
    
    def forward(self, x):
        n,m = x.shape
        assert m==self.m
        self.n = n
        
        self.x = x
        self.x_mean = np.mean(x,axis = 0)
        self.x_cent = x-self.x_mean
        self.x_square = self.x_cent**2
        self.x_var = np.mean(self.x_square,axis = 0)
        self.x_std = np.sqrt(self.x_var)
        self.x_frac_std = 1/self.x_std
        self.x_hat = self.x_cent*self.x_frac_std
        self.out = self.weight*self.x_hat+self.bias
        
        return self.out
    
    def backward(self, dout):
        # broadcast backward
        self._db += np.sum(dout, axis = 0)
        # dot product backward + broadcast backward
        self._dw += np.sum(self.x_hat*dout, axis = 0)
        # dot product backward
        self.dx_hat = self.weight*dout
        
        # dot product backward
        dx_cent1 = self.dx_hat*self.x_frac_std
        # dot product backward + broadcast backward
        dx_frac_std = np.sum(self.dx_hat*self.x_cent,axis=0)
        # divide backward
        dx_std = dx_frac_std*(-1/self.x_std**2)
        # sqrt backward
        dx_var = dx_std*(1/(2*self.x_std))
        # mean backward
        dx_square = np.tile(dx_var,[self.n,1])/self.n
        # square backward
        dx_cent2 = dx_square*(self.x_cent*2)
        # two sources
        dx_cent = dx_cent1+dx_cent2
        
        # add backward
        dx1 = dx_cent
        # minus backward + broadcast backward
        dx_mean = -np.sum(dx_cent,axis=0)
        # mean backward
        dx2 = np.tile(dx_mean,[self.n,1])/self.n
        # two sources
        dx = dx1+dx2
        
        return dx
    
    
    def zero_grad(self):
        self._db -= self._db
        self._dw -= self._dw
        
        
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = np.random.randn(num_embeddings, embedding_dim)
        self._dw = np.zeros_like(self.weight)
        self.state_dict['weight'] = {'value':self.weight,'grad':self._dw}
        
    def forward(self, x):
        '''
        x: np.array: dtype=int, shape=whatever
        return: np.array: dtype=float, shape=whatever+(embedding_dim,)
        '''
        self._x = x
        x_shape = x.shape
        x = x.flatten()
        out = self.weight[x]
        out = out.reshape(x_shape+(self.embedding_dim,))
        return out

    def backward(self, dz):
        '''
        dz: shape = whatever+(embedding_dim,)
        calculate gradient of weight
        hits: generally, indices is int type, so we cant get dx
        '''
        dz = dz.reshape(-1,self.embedding_dim)
        x = self._x.flatten()
        
        for i in range(len(x)):
            self._dw[x[i]] += dz[i]

        return None
        
    def zero_grad(self):
        self._dw -= self._dw
        
        
        
class UnPool2d(Module):
    def __init__(self, kernel_size = 2, stride = 2, padding=0):
        super().__init__()
        self._pool_h = kernel_size
        self._pool_w = kernel_size
        self._stride = stride
        self._pad = padding
        
        self._x = None
        self._arg_max = None

    def forward(self, x):
        self._x = x
        N, C, out_h, out_w = x.shape
        H = (out_h-1)*self._stride+self._pool_h-2*self._pad
        W = (out_w-1)*self._stride+self._pool_w-2*self._pad
        
        col_x = x.transpose(0,2,3,1).reshape(-1,1)
        col_x = np.tile(col_x,reps=(1,self._pool_h*self._pool_w))
        out = col2im(col_x, (N,C,H,W), self._pool_h,
                    self._pool_w, self._stride, self._pad)
        return out

    def backward(self, dz):
        N, C, out_h, out_w = self._x.shape
        H = (out_h-1)*self._stride+self._pool_h-2*self._pad
        W = (out_w-1)*self._stride+self._pool_w-2*self._pad
        dz = im2col(dz, self._pool_h, self._pool_w, self._stride, self._pad)
        dz = dz.reshape(N*out_h*out_w*C,-1)
        dz = np.sum(dz,axis = 1)
        dz = dz.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        return dz


##------------------------Neural Networks------------------------------------------------##











      
##------------------------DataLoader------------------------------------------------------##

from math import ceil

class DataLoader:
    def __init__(self, data, targets, batch_size, shuffle = False):
        indices = np.arange(data.__len__())
        if shuffle:
            np.random.shuffle(indices)
        self.data = data[indices]
        self.targets = targets[indices]
        self.batch_size = batch_size
        self.index = 0
        
    def get_batch(self):
        x = self.data[self.index:self.index+self.batch_size]
        y = self.targets[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        if self.index>=self.data.__len__():
            self.index = 0
        return x,y
    
    def __len__(self):
        return ceil(self.data.__len__()/self.batch_size)


##------------------------DataLoader------------------------------------------------------------##




















##------------------------Loss functions-------------------------------------------------------##

class CrossEntropyLossWithSoftMax:
    '''
    softmax，logits to probabilities
    using cross entropy loss to calculate gradient
    '''
    def __init__ (self, out_features):
        super().__init__()
        self.out_features = out_features
        
    def one_hot_encode(self, labels, num_classes):
        '''
        labels: 1 dim array
        return: one hot matrix 
        '''
        assert labels.max()+1<=num_classes
        
        batch_size = len(labels)
        oh_mat = np.zeros((batch_size,num_classes))
        oh_mat[range(batch_size),labels] = 1.
        return oh_mat
    
    def get_prob(self, x):
        '''
        calculate probabilities
        '''
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x,axis=1,keepdims=True)
        return exp_x/sum_exp_x
        
    def __call__(self, x, labels):
        '''
        cross entropy loss
        loss(l) = -log(p_l)
        '''
        batch_size = len(labels)
        y_hat = self.get_prob(x)
        loss = -np.mean(np.log(y_hat[range(batch_size),labels]))
        dx = y_hat-self.one_hot_encode(labels,self.out_features)
        return loss,dx/batch_size


class MSELoss:
    '''
    mean square loss
    '''
    def __init__ (self):
        super().__init__()
        
        
    def __call__(self, x, y):
        '''
        loss = (x-y)^2
        dL/dx = 2(x-y)
        '''
        batch_size = np.prod(x.shape)
        loss = ((x-y)**2).mean()
        dx = 2*(x-y)
        return loss,dx/batch_size


class BCELoss:
    '''
    bineay cross entropy loss
    '''
    def __init__ (self):
        super().__init__()
        
        
    def __call__(self, x, y):
        '''
        loss = -ylog(x)-(1-y)log(1-x)
        dL/dx = 2(x-y)
        '''
        batch_size = np.prod(x.shape)
        loss = (-y*np.log(x)-(1-y)*np.log(1-x)).mean()
        dx = -y/x+(1-y)/(1-x)
        return loss,dx/batch_size




##------------------------Loss functions-------------------------------------------------------##