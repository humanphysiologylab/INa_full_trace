__precompile__()

module INa

using DifferentialEquations: ODEProblem, solve
using DataFrames: DataFrame
using CSV: File as CSVFile


function find_step(t)
    index = findfirst(x -> x >= t, protocol.t)
    v = protocol.v[index]
end


function compute_algebraic(du, u, p, t)
    
    v_comp, v_p, v_m, m, h, j, I_out = u
    
    v_c = p.v_c  # find_step(t)
    
    tau_m = 1 / (p.a0_m * exp(v_m / p.s_m) + p.b0_m * exp(-v_m / p.delta_m))
    tau_h = 1 / (p.a0_h * exp(-v_m / p.s_h) + p.b0_h * exp(v_m / p.delta_h))
    tau_j = p.tau_j_const + 1 / (p.a0_j * exp(-v_m / p.s_j) + p.b0_j * exp(v_m / p.delta_j))
    
    m_inf = 1 / (1 + exp(-(p.v_half_m + v_m) / p.k_m))
    h_inf = 1 / (1 + exp((p.v_half_h + v_m) / p.k_h))
    
    v_cp = v_c + (v_c - v_comp) * (1 / (1 - p.alpha) - 1)

    I_leak = p.g_leak * v_m
    I_Na = p.g_max * h * m^3 * j * (v_m - p.v_rev)
    I_c = 1e9 * p.c_m * ((v_p + p.v_off - v_m) / (p.R * p.c_m) - 1e-9 * (I_leak + I_Na) / p.c_m)
    I_p = 1e9 * p.c_p * (v_cp - v_p) / (p.c_p * p.R_f)
    I_comp = 1e9 * p.x_c_comp * p.c_m * (v_c - v_comp) / (p.x_c_comp * p.c_m * p.x_r_comp * p.R * (1 - p.alpha))
    I_in = I_leak + I_Na + I_c  + I_p - I_comp
    
    a = (tau_m=tau_m, tau_h=tau_h, tau_j=tau_j,
         m_inf=m_inf, h_inf=h_inf,
         v_cp=v_cp, I_leak=I_leak, I_Na=I_Na, I_c=I_c, I_p=I_p, I_comp=I_comp, I_in=I_in,
         v_c=v_c)
    
    return a
end


function compute_rates!(du, u, p, t)
        
    v_comp, v_p, v_m, m, h, j, I_out = u
    a = compute_algebraic(du, u, p, t)
    
    du[1] = (a.v_c - v_comp) / (p.x_c_comp * p.c_m * p.x_r_comp * p.R * (1 - p.alpha))  # v_comp
    du[2] = (a.v_cp - v_p) / (p.c_p * p.R_f)  # v_p
    du[3] = (v_p + p.v_off - v_m) / (p.R * p.c_m) - (1e-9) * (a.I_leak + a.I_Na) / p.c_m  # v_m
        
    du[4] = (a.m_inf - m) / a.tau_m  # m
    du[5] = (a.h_inf - h) / a.tau_h  # h
    du[6] = (a.h_inf - j) / a.tau_j  # j
        
    du[7] = (a.I_in - I_out) / p.tau_z  # I_out
    
    nothing
end


function update_p(v_c, legend_constants)
    legend_constants[!, :value][end] = v_c
    p = collect(zip(Symbol.(legend_constants[!, :name]),
                legend_constants[!, :value]))
    p = NamedTuple(p)
end


function main()
    
    dirname_project = "/home/andrey/WORK/HPL/Code/INa_full_trace/"
    dirname_model = Base.Filesystem.joinpath(dirname_project, "src/model_ctypes/ina_pipette/")

    filename_legend_constants = Base.Filesystem.joinpath(dirname_model, "legend_constants.csv")
    filename_legend_states = Base.Filesystem.joinpath(dirname_model, "legend_test.csv")
    filename_protocol =  Base.Filesystem.joinpath(dirname_project, "data/protocols/protocol_sparse.csv")

    legend_states = DataFrame(CSVFile(filename_legend_states))
    legend_constants = DataFrame(CSVFile(filename_legend_constants))
    protocol = DataFrame(CSVFile(filename_protocol))
    
    tspan = (0., 0.)
    u₀ = [-80.0, -80.0, -80.0, 0.0, 1.0, 1.0, 0.0]
    du = zero(u₀)
    p = update_p(-80, legend_constants)

    # pl = plot()

    for row in eachrow(protocol[2:end, :])
        global tspan = (tspan[2], row.t)
        prob = ODEProblem(compute_rates!, u₀, tspan, p)
        sol = solve(prob, alg=Rodas5(), reltol=1e-8, abstol=1e-8)
    #     plot!(sol.t, sol[end, :], label=nothing)
        global p = update_p(row.v, legend_constants)
        global u₀ = deepcopy(sol.u[end])
    end

    println("DONE")
    
end

end # module
