#=
This file reads data files for modeling a unit commitment problem.

Aurhor: Kibaek Kim
=#
using DelimitedFiles

dBus = readdlm("$DATADIR/bus.dat", comments=true)
pBusType = dBus[:,2]

dLine = readdlm("$DATADIR/branch.dat", comments=true)
pFromBus = convert(Array{Int64,1},dLine[:,2])
pToBus = convert(Array{Int64,1},dLine[:,3])
pX = dLine[:,6]
pMaxFlow = dLine[:,8]

dPower = readdlm("$DATADIR/power_system_data.dat", comments=true)
pNumUnits = dPower[1]
pNumSegs = dPower[2]
# pNumSegs = 1

# pTime = dPower[3]
# pTime = 24

pNumBuses = dPower[4]
pNumLines = dPower[5]

sGen = 1:pNumUnits
sTime = 1:pTime
sSeg = 1:pNumSegs
sBus = 1:pNumBuses
sLine = 1:pNumLines

pGen2Bus = dPower[6,sGen]
pMaxGen = dPower[7,sGen]
pMinGen = dPower[8,sGen]
pInitState = dPower[10,sGen]
pMinDn = dPower[11,sGen]
pMinUp = dPower[12,sGen]
pRampUp = dPower[13,sGen]
pRampDn = dPower[14,sGen]
pOnCost = dPower[18,sGen]
pStart = dPower[19,sGen]
pCost = dPower[sGen.+22,sSeg]
pPowerRange = dPower[sGen .+ (22+pNumUnits), sSeg]

# 30BUS: 0.15
# RTS96: 0.45
# 118BUS: 0.45 ~ 0.5
dLoad = readdlm("$DATADIR/load.dat", comments=true) * pRatio
# if DATADIR == "./computation/UC_data/1354BUS"
#     dLoad = readdlm("$DATADIR/load.dat", comments=true)
# else
#     dLoad = CSV.read("./computation/UC_data/PJMestimated2016.txt", DataFrame)
#     dLoad = dLoad[sortperm(dLoad[:,1]),3]
#     dLoad = dLoad[sTime .+ (24*7*14)] * pRatio
# end
# @printf("Total Load: %f\n", sum(dLoad))
pDemand = dBus[:,3] / sum(dBus[:,3]) .* dLoad'

pDeg2Rad = pi/180
pBaseMVA = 100
pB = 1 ./pX * pDeg2Rad * pBaseMVA

pCommit = Dict()
pInitDn = zeros(Int64,pNumUnits)
pInitUp = zeros(Int64,pNumUnits)
for g = sGen
    pCommit[g] = 0
    if pInitState[g] < 0
        pInitDn[g] = max(0,pMinDn[g]+pInitState[g])
    elseif pInitState[g] > 0
        pInitUp[g] = max(0,pMinUp[g]-pInitState[g])
        pCommit[g] = 1
    end
end

pResvUp = 0.1
pResvDn = 0.05
pScale = 1.0e-3
