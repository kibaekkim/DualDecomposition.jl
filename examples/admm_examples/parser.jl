using ArgParse

function parse_commandline(s)
    @add_arg_table s begin
        "--dir"
            help = "output directory"
            arg_type = String
            default = "."
        # arguments for master
        "--master"
            help = "master algorithm:\n
                    -bm: bundle master\n
                    -am: admm master
                    "        
            arg_type = String
            default = "bm"
        "--timelim"
            help = "time limit"
            arg_type = Float64
            default = 3600.0
        "--tol"
            help = "tolerance level"
            arg_type = Float64
            default = 1e-6
        # arguments for sub
        "--mipsolver"
            help = "solver for subproblem:\n
                    -glpk\n
                    -highs\n
                    -cplex"
            arg_type = String
            default = "glpk"
        "--miptime"
            help = "time limit for mip"
            arg_type = Float64
            default = 3600.0
        "--mipgap"
            help = "gap tolerance for mip"
            arg_type = Float64
            default = 1.e-4
        "--miplog"
            help = "mip solver log"
            action = :store_true
        "--age"
            help = "cut age"
            arg_type = Int
            default = 10
        "--maxsubiter"
            help = "number of sub iterations"
            arg_type = Int
            default = 3000000
        # arguments for bundle method
        "--bmalg"
            help = "Bundle Method algorithm mode:\n
                    -0: proximal method" # TODO: other methods?
            arg_type = Int
            default = 0
        "--proxu"
            help = "initial proximal penalty value"
            arg_type = Float64
            default = 1.e-2
        "--numcut"
            help = "number of cuts"
            arg_type = Int
            default = 1
        # arguments for admm method
        "--amalg"
            help = "ADMM algorithm mode:\n
                    -0: constant ρ\n
                    -1: residual balancing\n
                    -2: adaptive residual balancing\n
                    -3: relaxed ADMM\n
                    -4: adaptive relaxed ADMM"
            arg_type = Int
            default = 1
        "--bundlelog"
            help = "admm subproblem bundle log"
            action = :store_true
        "--rho"
            help = "ADMM initial penalty value"
            arg_type = Float64
            default = 1.0
        "--tau"
            help = "ADMM Residual balancing multiplier"
            arg_type = Float64
            default = 2.0
        "--mu"
            help = "ADMM Residual balancing parameter"
            arg_type = Float64
            default = 1.0
        "--xi"
            help = "ADMM Residual balancing parameter"
            arg_type = Float64
            default = 10.0
        "--interval"
            help = "ADMM update interval"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end

parsed_args = parse_commandline(settings)

dir = parsed_args["dir"]

masteralg = parsed_args["master"]
timelim = parsed_args["timelim"]
tol = parsed_args["tol"]

mipsolver = parsed_args["mipsolver"]
if mipsolver == "cplex"
    using CPLEX
end
miptime = parsed_args["miptime"]
mipgap = parsed_args["mipgap"]
miplog = parsed_args["miplog"]
age = parsed_args["age"]
maxsubiter = parsed_args["maxsubiter"]

if masteralg == "bm"
    bmalg = parsed_args["bmalg"]
    proxu = parsed_args["proxu"]
    numcut = parsed_args["numcut"]
elseif masteralg == "am"
    amalg = parsed_args["amalg"]
    bundlelog = parsed_args["bundlelog"]
    rho = parsed_args["rho"]
    tau = parsed_args["tau"]
    mu = parsed_args["mu"]
    xi = parsed_args["xi"]
    uinterval = parsed_args["interval"]
end

