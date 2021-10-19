import numpy as np
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt

# @dataclass
# class diffusorParameters:

@dataclass
class SimulationProperties:
    distanceBetweenSoundSourceAndProbe: float   # r0
    distanceBetweenMicrophonesAndProbe: float   # r

@dataclass
class DiffusorProperties:
    algorithm: staticmethod
    modulo: int
    soundSpeed: float
    frequencyStart: float
    numberOfElements: int
    elementWidth: float
    angle: float
    samplesPerElement: int

    def createDiffusorSequence(self) -> list:
        return [((self.algorithm(x)) % self.modulo)
                * self.soundSpeed / self.frequencyStart / 2 / self.modulo
                for x in range(1, self.numberOfElements + 1)]

    def getMinimumWaveLength(self):
        return self.elementWidth / 2

    def calucalteK(self, frequency):
        return 2 * np.pi * frequency / self.soundSpeed


class Diffusor:
    def __init__(self, diffusorProperties: DiffusorProperties, diffusorAngleSequence: list):
        self.sequence = diffusorProperties.createDiffusorSequence()
        self.diffusorProperties = diffusorProperties
        self.diffusorAngleSequence = diffusorAngleSequence

        assert len(sequence) == len(diffusorAngleSequence)

        self.renderDiffusorFunction()

    def createRamp(self, width: float, baseLevel: float, side: str, n: int):
       const = np.tan(self.diffusorProperties.angle)
       tempY = list()
       for index in range(n):
           tempX = index * width / n
           tempY.append(const * tempX)
       tempY = np.array(tempY) if side == 'L' else np.flip(np.array(tempY))
       return baseLevel + tempY

    def renderDiffusorFunction(self):
        self.y = np.array([])
        for index, level in enumerate(self.sequence):
            self.y = np.append(self.y, self.createRamp(
                self.diffusorProperties.elementWidth,
                level,
                self.diffusorAngleSequence[index],
                self.diffusorProperties.samplesPerElement))

        self.x = (np.arange(len(self.y)) / len(self.y) - 0.5) * \
            self.diffusorProperties.elementWidth * self.diffusorProperties.numberOfElements
        print("Diffusor function was rendered!")

def angles():
    return np.arange(0, np.pi, np.pi/180)

def R(k, currentHeight):
    return np.exp(-1j * k * 2 * currentHeight)


def alpha(k, r, r0):
    return -1j * k / (8 * np.pi ** 2) * np.exp(-1j * k * (r + r0))


def beta(k, b, r, phi):
    return np.sinc(k * b / r) * (np.cos(phi + 1))

def calculateIntegral(x: np.array, values: list) -> float:
    sum = 0
    for xIndex in range(1, len(x)):
        currentXHeight = abs(x[xIndex] - x[xIndex-1])
        tempSurface = 1 / 2 * (values[xIndex] + values[xIndex-1]) * currentXHeight
        sum += tempSurface
    return sum


def gamma(k, x: np.array, phi, y: np.array):
    values = list()
    for index, singleX in enumerate(x):
        temp = R(k, y[index]) * np.exp(1j * k * singleX * np.sin(phi))
        values.append(temp)
    return calculateIntegral(x, values)


if __name__ == "__main__":
    diffusorProperties = DiffusorProperties(
        algorithm=lambda x: x**5,
        modulo=23,
        soundSpeed=343.6,
        frequencyStart=500,
        numberOfElements=46,
        elementWidth=0.021739,
        angle=np.pi/6,
        samplesPerElement=30)

    simulationProperties = SimulationProperties(10, 5)

    sequence = diffusorProperties.createDiffusorSequence()

    diffusorAngleSequence = ['L', 'R', 'L', 'R', 'R', 'R', 'R', 'R',
                             'L', 'L', 'R', 'L', 'R', 'L', 'R', 'R',
                             'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L',
                             'R', 'L', 'L', 'L', 'R', 'L', 'L', 'R',
                             'R', 'L', 'L', 'L', 'R', 'R', 'R', 'L',
                             'R', 'L', 'L', 'R', 'L', 'L']

    diffusor = Diffusor(diffusorProperties, diffusorAngleSequence)

    frequencies = [100,  125,  160,  200,  250,
                   315,  400,  500,  630,  800,
                   1000, 1250, 1600, 2000, 2500]

    results = dict() 
    for freq in frequencies:
        print(f"calculating frequency {freq} Hz")
        k = diffusorProperties.calucalteK(freq)
        firstPart = alpha(k, simulationProperties.distanceBetweenSoundSourceAndProbe,
                         simulationProperties.distanceBetweenSoundSourceAndProbe)

        sizeOfDiffusor = max(diffusor.x)
        
        phases = dict()
        for phase in np.arange(0, np.pi, np.pi/180):
            second = beta(k, sizeOfDiffusor, simulationProperties.distanceBetweenMicrophonesAndProbe, phase)
            third = gamma(k, diffusor.x, phase, diffusor.y) 
            phases[phase] = np.real(np.abs(firstPart * second * third))

        results[freq] = phases
    
    endResult = {"values": list(), "frequencies": list(), "parameterName" : "Acoustic Diffusion Coefficient"}
    for freq in results.keys():
        tempResult = list(results[freq].values())
        print(tempResult)
        first = np.square(np.sum(tempResult))
        second = np.sum(np.square(tempResult))
        third = (len(tempResult) - 1) * second
        endResult["frequencies"].append(freq)
        endResult["values"].append((first - second) / third)

    print(json.dumps(endResult, indent=3))
        
    plt.plot(endResult["frequencies"], endResult["values"])
    plt.show()
