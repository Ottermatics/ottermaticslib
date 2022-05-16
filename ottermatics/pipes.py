

'''We'll use the QP formulation to develop a fluid analysis system for fluids
start with single phase and move to others


1) Pumps Power = C x Q x P 
2) Compressor = C x (QP) x (PR^C2 - 1)
3) Pipes = dP = C x fXL/D x V^2 / 2g
4) Pipe Fittings / Valves = dP = C x V^2 (fXL/D +K)   |   K is a constant, for valves it can be interpolated between closed and opened
5) Splitters & Joins: Handle fluid mixing
6) Phase Separation (This one is gonna be hard)
7) Heat Exchanger dP = f(V,T) - phase change issues
8) Filtration dP = k x Vxc x (Afrontal/Asurface) "linear"
'''