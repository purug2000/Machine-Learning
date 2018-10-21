import numpy as np
import random
import cPickle
import gzip
class Network(object):      ##A neural Net 
    def __init__(self,LayerSizes,filename=None):
        if filename==None:
            self.num_layers=len(LayerSizes)
            self.LayerSizes=LayerSizes
            self.biases=[np.random.randn(numNeuron,1) for numNeuron in LayerSizes[1:]]
            self.weights=[np.random.randn(y,x) for x,y in zip(LayerSizes[:-1],LayerSizes[1:])]
        else:
            [self.weights,self.biases]=np.load(filename)
            self.num_layers=len(self.biases)+1
            self.LayerSizes=[len(w) for w in self.weights]+[LayerSizes[-1]]
            
    def perceptron(self,z,bound):
        if z>bound:
            return 1
        else:
            return 0
    def feedforward(self,a):
        ##Find the output of neural net given an input a 
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
    def SGD(self, training_data,epochs,mbs,eta,test_data=None):
        if test_data:
            n_t=len(test_data)
        n=len(training_data)
        for k in xrange(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[t:t+mbs] for t in xrange(0,n,mbs) ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print "Epoch {}: {}/{}".format(k,self.evaluate(test_data),n_t)
            else:
                print "Epoch {} complete".format(k)

    def update_mini_batch(self,mini_batch,eta):
        new_b=[np.zeros(b.shape) for b in self.biases]
        new_w= [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            change_b,change_w=self.backprop(x,y)
            new_b=[nb+cnb for nb,cnb in zip(new_b,change_b)]
            new_w=[nw+cnw for nw,cnw in zip(new_w,change_w)]
            self.weights=[w-eta/len(mini_batch)*nw for nw, w in zip(new_w,self.weights)]
            self.biases=[b-eta/len(mini_batch)*nb for nb, b in zip(new_b,self.biases)]
    def backprop(self, x,y):
        new_b=[np.zeros(b.shape) for b in self.biases]
        new_w= [np.zeros(w.shape) for w in self.weights]
        layeroutput=x
        layeroutputs=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,layeroutput)+b
            zs.append(z)
            layeroutput=sigmoid(z)
            layeroutputs.append(layeroutput)
        d=self.cost_derivative(layeroutputs[-1],y)*grad_sigmo(zs[-1])
        new_b[-1]=d
        new_w[-1]=np.dot(d,layeroutputs[-2].transpose())
        for l in range(2,self.num_layers):
            z=zs[-l]
            gs=grad_sigmo(z)
            d=np.dot(self.weights[-l+1].transpose(),d)*gs  
            new_b[-l]=d
            new_w[-l]=np.dot(d,layeroutputs[-l-1].transpose())   
        return (new_b,new_w)

    def cost_derivative(self,a,y):
        return (a-y)
    def evaluate(self, test_data):
        test_results=[(np.argmax(self.feedforward(x)),y) for x,y in test_data]
        return sum(int(x==y) for  x,y in test_results)
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
def grad_sigmo(z):
    return sigmoid(z)*(1-sigmoid(z))
def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

training_data, validation_data, test_data=load_data_wrapper()

for j in range(4):
    net=Network([784,30,10],'NNet.npy')
    net.SGD(training_data,1,10,3.0,test_data=test_data,)
    x=[net.weights ,net.biases]
    np.save('NNet',x)
    x=np.load('NNet.npy')
    #print x
