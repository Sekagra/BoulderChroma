# ![icon](https://i.imgur.com/CzteKMb.png) &nbsp;&nbsp;&nbsp;&nbsp;  BoulderChroma


Detection of colors of boulders for color-blind people.

## Inspiration
Indoor bouldering is a form of climbing sport that is performed on small artificial rock walls without the use of ropes or harnesses for safety. A bouldering route consists of a set of multiple holds with a defined start and end hold. Walls in indoor gyms are set with different boulders in varying difficulties at the same time. To distinguish between the individual routes, holds that are part of the same route have the same color.

A frequent problem which arose in our team's past bouldering sessions was the inability of one color-blind team member to recognize which holds are part of which route. Especially between red and green routes or variations of these colors, color-blind people can have difficulties to tell the holds of different routes apart.

## What it does
Our app provides a solution by using augmented reality to enhance the live camera feed of a smart phone. The app recognizes boulder holds in the camera feed and augments them with a bounding box and a color label.

## How I built it
Using Cognitive Services in Microsoft Azure, we used photos of boulder walls with colored routes (mostly from our TUM bouldering egg) to train a neural network for detecting bouldering holds and their colors.

## Impressions

![](https://i.imgur.com/NVFq5Hv.jpg)
![](https://i.imgur.com/R3W8yOi.jpg)
![](https://i.imgur.com/RLYy8uN.jpg)
![](https://i.imgur.com/fySklFE.jpg)
