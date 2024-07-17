using DualDecomposition
using LinearAlgebra
using JuMP
using Random

using ArgParse

const DD = DualDecomposition
const parallel = DD.parallel

settings = ArgParseSettings()
@add_arg_table settings begin
    "--DataDir"
        help = "directory of data"
        arg_type = String
        required = true
    "--GraphDir"
        help = "directory of graph"
        arg_type = String
        default = ""
    "--pRatio"
        help = "pRatio"
        arg_type = Float64
        default = 0.5
    "--pPenalty"
        help = "penalty for slack/surplus on flow balance constraint"
        arg_type = Float64
        default = 1e5
    "--nStage"
        help = "number of stages"
        arg_type = Int
        default = 4
    "--nPeriod"
        help = "number of time periods per stage"
        arg_type = Int
        default = 6
    "--nBranch"
        help = "number of scenarios per stage"
        arg_type = Int
        default = 2
    "--wResv"
        help = "change model to using resv"
        action = :store_true
    "--decmode"
        help = "decomposition mode:\n
                -0 no decomposition\n
                -1 scenario decomposition\n
                -2 nodal decomposition
                "
        arg_type = Int
        default = 1
end

include("../parser.jl")

DATADIR = parsed_args["DataDir"]
GRAPHDIR = parsed_args["GraphDir"]
pRatio = parsed_args["pRatio"]
pPenalty = parsed_args["pPenalty"]
nStage = parsed_args["nStage"]
nPeriod = parsed_args["nPeriod"]
pTime = nStage * nPeriod

nBranch = parsed_args["nBranch"]
wResv = parsed_args["wResv"]

decmode = parsed_args["decmode"]

mkpath(dir)

include("uc_data.jl")

if GRAPHDIR == ""
    GRAPHDIR = dir
end
include("createTree.jl")

@assert height(LoadTree) + 1 == pTime
checknode = LoadTree.children[1][1]
for t in 1:nPeriod
    if t < nPeriod
        @assert length(LoadTree.children[checknode+1]) == 1
        global checknode = LoadTree.children[checknode+1][1]
    else
        @assert length(LoadTree.children[checknode+1]) == nBranch
    end
end

scentree_block = Dict()

function create_node!(tree::DD.Tree, pt::Int, pid::Int, hist::Vector{Tuple{Int,Vector{Int}}})
    #tree:  DD scenario tree
    #pt:    DD parent id
    #pid:   scentree parent id
    #hist:  scentree id history vector of (DD id, [scentree id])
    for s = LoadTree.children[pid+1]
        prob = LoadTree.probability[s]
        node_set = Vector{Int}()
        Load = Vector{Float64}()
        current_node = s
        for t in 1:nPeriod
            push!(node_set, current_node)
            push!(Load, LoadTree.state[current_node])
            if t < nPeriod
                current_node = LoadTree.children[current_node+1][1]
            end
        end
        ξ = Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(:Load => Load)
        if pt == 0
            DD.add_node!(tree, DD.TreeNode(ξ))
            id = 1
        else
            id = DD.add_child!(tree, pt, ξ, prob)
        end
        scentree_block[id] = copy(node_set)
        function subproblem_builder(tree::DD.Tree, subtree::DD.SubTree, node::DD.SubTreeNode)
            m = subtree.model
            Load = DD.get_scenario(node)[:Load]

            pDemandNode = dBus[:,3] / sum(dBus[:,3]) .* Load' * pRatio
            stages = ScenTrees.stage(LoadTree, node_set)

            vCommit = @variable(m, vCommit[g=sGen,t=1:nPeriod], Bin, base_name="n$(id)_vCommit")
            if stages[end] + 1 < pTime
                for g=sGen
                    DD.set_output_variable!(node, Symbol("vCommit_$g"), vCommit[g,end])
                end
            end

            vStart = @variable(m, 0 <= vStart[g=sGen,t=1:nPeriod] <= 1, base_name="n$(id)_vStart")
            if stages[end] + 1 < pTime
                for g=sGen
                    for q = 1:min(pMinUp[g]-1,nPeriod)
                        t = node_set[end-q+1]
                        DD.set_public_output_variable!(node, id, Symbol("vStart_$(g)_$t"), vStart[g,end-q+1])
                    end
                end
            end

            vDown = @variable(m, 0 <= vDown[g=sGen,t=1:nPeriod] <= 1, base_name="n$(id)_vDown")
            if stages[end] + 1 < pTime
                for g=sGen
                    for q = 1:min(pMinDn[g]-1,nPeriod)
                        t = node_set[end-q+1]
                        DD.set_public_output_variable!(node, id, Symbol("vDown_$(g)_$t"), vDown[g,end-q+1])
                    end
                end
            end

            vGen = @variable(m, 0 <= vGen[g=sGen,t=1:nPeriod] <= pMaxGen[g], base_name="n$(id)_vGen")
            if stages[end] + 1 < pTime
                for g=sGen
                    DD.set_output_variable!(node, Symbol("vGen_$g"), vGen[g,end])
                end
            end
            if wResv
                vResvUp = @variable(m, vResvUp[g=sGen,t=1:nPeriod], base_name="n$(id)_vResvUp")
                vResvDn = @variable(m, vResvDn[g=sGen,t=1:nPeriod], base_name="n$(id)_vResvDn")
            end
            @variable(m, 0 <= vSeg[g=sGen,k=sSeg,t=1:nPeriod] <= pPowerRange[g,k], base_name="n$(id)_vSeg")
            @variable(m, -pMaxFlow[l] <= vFlow[l=sLine,t=1:nPeriod] <= pMaxFlow[l], base_name="n$(id)_vFlow")
            @variable(m, -180 <= vAngle[n=sBus,t=1:nPeriod] <= 180, base_name="n$(id)_vAngle")
            @variable(m, vSlack[n=sBus,t=1:nPeriod]>=0, base_name="n$(id)_vSlack")
            @variable(m, vSplus[n=sBus,t=1:nPeriod]>=0, base_name="n$(id)_vSplus")

            if stages[1] > 0
                _vCommit = @variable(m, 0 <= _vCommit[g=sGen] <= 1, base_name="n$(id)__vCommit")
                for g=sGen
                    DD.set_input_variable!(node, Symbol("vCommit_$g"), _vCommit[g])
                end
            end

            vStartT = @variable(m, 0 <= vStartT[g=sGen,q=1:pMinUp[g]-1] <= 1, base_name="n$(id)_vStartT")
            for g=sGen
                for q=1:pMinUp[g]-1
                    if q > length(hist) * nPeriod
                        @constraint(m, vStartT[g,q] == 0)
                    else
                        (source, nodels) = hist[end-(q-1)÷nPeriod]
                        t = nodels[end-(q-1)%nPeriod]
                        DD.set_public_input_variable!(node, source, Symbol("vStart_$(g)_$t"), vStartT[g,q])
                    end
                end
            end

            vDownT = @variable(m, 0 <= vDownT[g=sGen,q=1:pMinDn[g]-1] <= 1, base_name="n$(id)_vDownT")
            for g=sGen
                for q=1:pMinDn[g]-1
                    if q > length(hist) * nPeriod
                        @constraint(m, vDownT[g,q] == 0)
                    else
                        (source, nodels) = hist[end-(q-1)÷nPeriod]
                        t = nodels[end-(q-1)%nPeriod]
                        DD.set_public_input_variable!(node, source, Symbol("vDown_$(g)_$t"), vDownT[g,q])
                    end
                end
            end

            if stages[1] > 0
                _vGen = @variable(m, 0 <= _vGen[g=sGen] <= pMaxGen[g], base_name="n$(id)__vGen")
                for g=sGen
                    DD.set_input_variable!(node, Symbol("vGen_$g"), _vGen[g])
                end
            end

            @constraint(m, eBalance[n=sBus,t=1:nPeriod],
                sum(vFlow[l,t] for l = sLine if pToBus[l] == n) 
                - sum(vFlow[l,t] for l = sLine if pFromBus[l] == n) 
                + sum(vGen[g,t] for g = sGen if pGen2Bus[g] == n) + vSlack[n,t] - vSplus[n,t]
                == pDemandNode[n,t])

            @constraint(m, eFlow[l=sLine,t=1:nPeriod],
                vFlow[l,t] == pB[l] * (vAngle[pFromBus[l],t] - vAngle[pToBus[l],t]))

            @constraint(m, eGen[g=sGen,t=1:nPeriod], vGen[g,t] == pMinGen[g] * vCommit[g,t] + sum(vSeg[g,k,t] for k = sSeg))
            
            if wResv
                @constraint(m, eMinGen[g=sGen,t=1:nPeriod], vGen[g,t] >= vResvDn[g,t])
                @constraint(m, eMaxGen[g=sGen,t=1:nPeriod], vGen[g,t] <= vResvUp[g,t])
                @constraint(m, eMinResv[g=sGen,t=1:nPeriod], vResvDn[g,t] >= pMinGen[g] * vCommit[g,t])
                @constraint(m, eMaxResv[g=sGen,t=1:nPeriod], vResvUp[g,t] <= pMaxGen[g] * vCommit[g,t])
            else
                @constraint(m, eMinGen[g=sGen,t=1:nPeriod], vGen[g,t] >= pMinGen[g] * vCommit[g,t])
                @constraint(m, eMaxGen[g=sGen,t=1:nPeriod], vGen[g,t] <= pMaxGen[g] * vCommit[g,t])
            end

            if wResv
                @constraint(m, eResvUp[t=1:nPeriod],
                    sum(vResvUp[g,t] for g = sGen) >= (1+pResvUp) * sum(pDemand[n,t] for n = sBus))
                @constraint(m, eResvDn[t=1:nPeriod],
                    sum(vResvDn[g,t] for g = sGen) <= (1-pResvDn) * sum(pDemand[n,t] for n = sBus))

                if stages[1] > 0
                    @constraint(m, eRampUp1[g=sGen], vResvUp[g,1] - _vGen[g] <= pRampUp[g] * _vCommit[g] + pMaxGen[g] * vStart[g,1])
                end
                @constraint(m, eRampUp[g=sGen,t=2:nPeriod], vResvUp[g,t] - vGen[g,t-1] <= pRampUp[g] * vCommit[g,t-1] + pMaxGen[g] * vStart[g,t])
                if stages[1] > 0
                    @constraint(m, eRampDn1[g=sGen], vResvDn[g,1] - _vGen[g] >= -pRampDn[g] * _vCommit[g] - pMaxGen[g] * vDown[g,1])
                end
                @constraint(m, eRampDn[g=sGen,t=2:nPeriod], vResvDn[g,t] - vGen[g,t-1] >= -pRampDn[g] * vCommit[g,t-1] - pMaxGen[g] * vDown[g,t])
            else
                if stages[1] > 0
                    @constraint(m, eRampUp1[g=sGen], vGen[g,1] - _vGen[g] <= pRampUp[g] * _vCommit[g] + pMaxGen[g] * vStart[g,1])
                end
                @constraint(m, eRampUp[g=sGen,t=2:nPeriod], vGen[g,t] - vGen[g,t-1] <= pRampUp[g] * vCommit[g,t-1] + pMaxGen[g] * vStart[g,t])
                if stages[1] > 0
                    @constraint(m, eRampDn1[g=sGen], vGen[g,1] - _vGen[g] >= -pRampDn[g] * _vCommit[g] - pMaxGen[g] * vDown[g,1])
                end
                @constraint(m, eRampDn[g=sGen,t=2:nPeriod], vGen[g,t] - vGen[g,t-1] >= -pRampDn[g] * vCommit[g,t-1] - pMaxGen[g] * vDown[g,t])
            end

            # Initial up and down
            @constraint(m, eInitUp[g=sGen,t=1:pInitUp[g]; stages[t] + 1 <= pInitUp[g]], vCommit[g,t] == 1)
            @constraint(m, eInitDn[g=sGen,t=1:pInitDn[g]; stages[t] + 1 <= pInitDn[g]], vCommit[g,t] == 0)
            
            @constraint(m, eMinUp2[g=sGen,t=1:nPeriod], 
                sum(vStartT[g,q] for q in 1:(pMinUp[g]-t)) + sum(vStart[g,q] for q= max(1,t-pMinUp[g]+1):t) <= vCommit[g,t])
            @constraint(m, eMinDn2[g=sGen,t=1:nPeriod], 
                sum(vDownT[g,q]  for q in 1:(pMinDn[g]-t)) + sum(vDown[g,q]  for q= max(1,t-pMinDn[g]+1):t) <= 1 - vCommit[g,t])

            if stages[1] > 0
                @constraint(m, eDown1[g=sGen], vDown[g,1] <= _vCommit[g])
                @constraint(m, eStart1[g=sGen], vStart[g,1] <= 1 - _vCommit[g])
                @constraint(m, eCommit1[g=sGen], vStart[g,1] - vDown[g,1] == vCommit[g,1] - _vCommit[g])
            else
                @constraint(m, eDown0[g=sGen], vDown[g,1] <= pCommit[g])
                @constraint(m, eStart0[g=sGen], vStart[g,1] <= 1 - pCommit[g])
                @constraint(m, eCommit0[g=sGen], vStart[g,1] - vDown[g,1] == vCommit[g,1] - pCommit[g])
            end
            @constraint(m, eDown[g=sGen,t=2:nPeriod], vDown[g,t] <= vCommit[g,t-1])
            @constraint(m, eStart[g=sGen,t=2:nPeriod], vStart[g,t] <= 1 - vCommit[g,t-1])
            @constraint(m, eCommit[g=sGen,t=2:nPeriod], vStart[g,t] - vDown[g,t] == vCommit[g,t] - vCommit[g,t-1])

            @constraint(m, eRefBus[n=sBus,t=1:nPeriod; pBusType[n] == 3], vAngle[n,t] == 0)

            DD.set_stage_objective(node, pScale * sum(pOnCost[g] * vCommit[g,t] for g = sGen for t = 1:nPeriod)
                + pScale * sum(pStart[g] * vStart[g,t] for g = sGen for t = 1:nPeriod)
                + pScale * sum(pCost[g,k] * vSeg[g,k,t] for g = sGen for k = sSeg for t = 1:nPeriod)
                + pScale * pPenalty * sum(vSlack[n,t] + vSplus[n,t] for n = sBus for t = 1:nPeriod))
            
            for (key,var) in JuMP.object_dictionary(m)
                JuMP.unregister(m, key)
            end
        end
        DD.set_stage_builder!(tree, id, subproblem_builder)
        if DD.get_stage(tree, id) * nPeriod < pTime
            newhist = copy(hist)
            push!(newhist, (id, node_set))
            create_node!(tree, id, node_set[end], newhist)
        end
    end
end

function create_root()
    tree = DD.Tree()

    hist = Vector{Tuple{Int,Vector{Int}}}()
    create_node!(tree, 0, 0, hist)
    return tree
end

tree = create_root()
if decmode == 0
    node_cluster = DD.decomposition_not(tree)
elseif decmode == 1
    node_cluster = DD.decomposition_scenario(tree)
elseif decmode == 2
    node_cluster = DD.decomposition_temporal(tree)
end
NS = length(node_cluster)

function create_sub_model!(block_id::Int64, coupling_variables::Vector{DD.CouplingVariableRef})
    nodes = node_cluster[block_id]
    subtree = DD.create_subtree!(tree, block_id, coupling_variables, nodes)
    return subtree.model
end

include("../core.jl")
