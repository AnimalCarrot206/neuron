import numpy as numpy

# Функция активации: f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
  return 1 / (1 + numpy.exp(-x))

# Класс нейрона
class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias
  # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
  def feedforward(self, inputs):
    total = numpy.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

class OurNeuralNetwork:
  '''
  Нейронная сеть с:
    - 2 входами
    - скрытым слоем с 2 нейронами (h1, h2)
    - выходным слоем с 1 нейроном (o1)
  Все нейроны имеют одинаковые веса и пороги:
    - w = [0, 1]
    - b = 0
  '''
  def __init__(self):
    weights = numpy.array([0, 1])
    bias = 0

    # Используем класс Neuron
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # Входы для o1 - это выходы h1 и h2
    out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))

    return out_o1

network = OurNeuralNetwork()
x = numpy.array([5, 6])
print(network.feedforward(x))