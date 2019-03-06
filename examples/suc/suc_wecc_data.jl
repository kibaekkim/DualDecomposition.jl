using DelimitedFiles

mutable struct UnitCommitmentModel

    # Sets
    IMPORT # import points
    G      # generators
    Gf     # fast generators
    Gs     # slow generators
    L      # transmission lines
    LOAD   # loads
    N      # buses
    RE     # renewable generators
    T      # time periods
    T0     # 0..|T|
    WIND   # wind farms

    # Cost parameters
    C  # generation cost
    Cl # loadsheding cost
    Ci # import spillage cost
    Cr # renewable spillage cost
    Cw # wind spillage cost
    K  # commitment cost
    S  # startup cost

    # Capacity parameters
    B  # line susceptance
    Pmax # max generation capacity
    Pmin # min generation capacity
    Rmax # max ramping capacity
    Rmin # min ramping capacity
    TC   # transmission line capacity
    DT   # minimum downtime of generator g
    UT   # minimum uptime of generator g

    # Supply/demand parameters
    D    # netload in bus n, time t, scenarion j
    Igen # generation from import points
    Rgen # generation from renewable
    Wgen # wind generation
    load # load at load i at time t

    # Mapping parameters
    gen2bus    # map generator to bus
    import2bus # map import point to bus
    load2bus   # map load to bus
    re2bus     # map renewable generator to bus
    wind2bus   # map wind farm to bus
    fbus       # bus from which line l flows
    tbus       # bus to which line l flows

    π # probability

    UnitCommitmentModel() = new()
end

function weccdata(nScenarios::Integer, offset::Integer, Season::AbstractString)::UnitCommitmentModel

    # read file and create dictionary
    function readDict(f)
        d = readdlm(f);
        return Dict(zip(d[:,1], d[:,2]));
    end

    # read file and create dictionary in reverse column indices
    function readRDict(f)
        d = readdlm(f);
        return Dict(zip(d[:,2], d[:,1]));
    end

    # create a dictionary for 2-dimensional data
    function create2DDict(d)
        dd = Dict();
        for j in 2:size(d,2)
            for t = 1:24
                dd[d[1,j],t] = d[t+1,j]
            end
        end
        return dd;
    end

    # Set paths
    DATA_DIR = "./examples/suc/WECC_DATA";
    WIND_DIR = "$DATA_DIR/WIND/$Season";

    uc = UnitCommitmentModel()

    # ---------------
    # Read data files
    # ---------------

    # BUSES
    uc.N = readdlm("$DATA_DIR/Buses.txt");                     # list of buses
    uc.gen2bus = readDict("$DATA_DIR/BusGenerators.txt");      # bus to generators
    uc.import2bus = readDict("$DATA_DIR/BusImportPoints.txt"); # bus to import points
    uc.load2bus = readDict("$DATA_DIR/BusLoads.txt");          # bus to loads
    uc.re2bus = readDict("$DATA_DIR/BusREGenerators.txt");     # bus to RE generators
    uc.wind2bus = readDict("$DATA_DIR/BusWindGenerators.txt"); # bus t wind generators

    # Generators
    uc.G = readdlm("$DATA_DIR/Generators.txt");         # list of generators
    fastgen = readDict("$DATA_DIR/FastGenerators.txt")  # fast generators
    uc.Gs = AbstractString[]
    uc.Gf = AbstractString[]
    for g in uc.G
        if fastgen[g] == "y"
            push!(uc.Gf,g)
        else
            push!(uc.Gs,g)
        end
    end
    uc.Pmax = readDict("$DATA_DIR/MaxRunCapacity.txt"); # max generation capacity
    uc.Pmin = readDict("$DATA_DIR/MinRunCapacity.txt"); # min generation capacity
    uc.Rmin = readDict("$DATA_DIR/RampDown.txt");       # ramp down limit
    uc.Rmax = readDict("$DATA_DIR/RampUp.txt");         # ramp up limit
    uc.C = readDict("$DATA_DIR/FuelPrice.txt");         # generation cost
    uc.K = readDict("$DATA_DIR/C0.txt");                # operating cost
    uc.S = readDict("$DATA_DIR/SUC.txt");               # start-up cost
    uc.UT = readDict("$DATA_DIR/UT.txt");               # minimum uptime
    uc.DT = readDict("$DATA_DIR/DT.txt");               # minimum downtime
    # @show length(uc.G)
    # @show length(uc.Gs)
    # @show length(uc.Gf)

    # Calculated Netdemand load
    uc.LOAD = readdlm("$DATA_DIR/Loads.txt"); # list of loads
    tmp = readdlm("$DATA_DIR/Demand$Season.txt")
    uc.load = create2DDict(tmp);
    # @show length(uc.LOAD)

    # IMPORTS
    uc.IMPORT = readdlm("$DATA_DIR/ImportPoints.txt");
    tmp = readdlm("$DATA_DIR/ImportProduction$Season.txt")
    uc.Igen = create2DDict(tmp);
    # @show length(uc.IMPORT)

    # Non-wind renewable production
    uc.RE = readdlm("$DATA_DIR/REGenerators.txt");
    tmp = readdlm("$DATA_DIR/REProduction$Season.txt")
    uc.Rgen = create2DDict(tmp);
    # @show length(uc.RE)

    # Network
    uc.L = readdlm("$DATA_DIR/Lines.txt"); # list of lines
    uc.fbus = readDict("$DATA_DIR/FromBus.txt");
    uc.tbus = readDict("$DATA_DIR/ToBus.txt");
    uc.TC = readDict("$DATA_DIR/TC.txt"); # line capacity
    uc.B = readDict("$DATA_DIR/Susceptance.txt");

    # WINDS
    uc.WIND = readdlm("$DATA_DIR/WindGenerators.txt"); # list of wind generators
    dWindProductionSamples = readdlm("$WIND_DIR/WindProductionSamples.txt");
    # @show length(uc.WIND)

    # ADDITIONAL PARAMETERS
    penetration = 0.1
    nPeriods = 24
    uc.π = ones(nScenarios) / nScenarios # equal probabilities
    uc.Cl = 5000 # value of lost load ($/MWh)
    uc.Ci = 0 # import spillage penalty
    uc.Cr = 0 # renewable spillage penalty
    uc.Cw = 0 # wind spillage penalty

    # ADDITIONAL SETS
    uc.T0 = 0:nPeriods;
    uc.T = 1:nPeriods;

    # Wind production scenarios
    # TODO: not really generic
    reshapedWind = reshape(dWindProductionSamples[2:24001,2:6], nPeriods, 1000, length(uc.WIND));
    uc.Wgen = Dict();
    for w in 1:length(uc.WIND)
        for t in uc.T
            for s in 1:nScenarios
                # This calculates the production level scaled by a given penetration.
                uc.Wgen[uc.WIND[w],t,s] = reshapedWind[t,offset+s,w] * penetration / 0.15;
            end
        end
    end

    uc.D = Dict();
    for n = uc.N
        for t = uc.T
            nd = 0.0;
            for j in uc.LOAD
                if uc.load2bus[j] == n
                    nd += uc.load[j,t];
                end
            end
            for j in uc.IMPORT
                if uc.import2bus[j] == n
                    nd -= uc.Igen[j,t];
                end
            end
            for j in uc.RE
                if uc.re2bus[j] == n
                    nd -= uc.Rgen[j,t];
                end
            end
            uc.D[n,t] = nd
        end
    end

    return uc
end
