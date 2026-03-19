# Module 03: Unreal Engine

## Project Files

UE5 project files exceed 100MB and cannot be hosted on GitHub.

**OneDrive Links:**
- [Niagara_test2.zip (UE Project)](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucbv512_ucl_ac_uk/IQAcYDZaTZBETae4gxXVY24-AcoZLthi7GDia0bszxxIPM4?e=0UsAl3)
- [ARCGIS.zip (GIS Project Files)](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucbv512_ucl_ac_uk/IQDIDfrpRPtwT5bGW-8U2O0MAcz4RAC6Ww1Hy1svnYwgE58?e=ncj8z1)

**Video:** [UE Niagara Systems Demo (YouTube)](https://youtu.be/oyOwx5gqYzE)

## Contents

### Niagara System 1: Industrial Smoke Plume (05_smoke1)
- SimpleSpriteBurst emitter
- Key modules: Initialize Particle, System Location, Add Velocity Linear, Scale Color, Drag, Scale Sprite Size, Sub UV Animation
- Material: T_Smoke_SubUV_01_Mat (Translucent, SubUV flipbook)

### Niagara System 2: Spark/Ember Burst (04_spark1)
- SimpleSpriteBurst emitter, 500 particles
- Key modules: Shape Location Sphere, Add Velocity In Cone, Color from Curve, Gravity Force, Curl Noise Force, Collision (Ray Traced)
- Material: flame02_Mat (Translucent, x100 Emissive for HDR bloom)

## Software
- Unreal Engine 5.4
- Niagara Visual Effects System
