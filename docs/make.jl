using LinearAlgebraUtils
using Documenter

DocMeta.setdocmeta!(LinearAlgebraUtils, :DocTestSetup, :(using LinearAlgebraUtils); recursive=true)

makedocs(;
    modules=[LinearAlgebraUtils],
    authors="Shobhan Kulshreshtha <skulshre@usc.edu",
    repo="https://github.com/shobhan126/LinearAlgebraUtils.jl/blob/{commit}{path}#{line}",
    sitename="LinearAlgebraUtils.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://shobhan126.github.io/LinearAlgebraUtils.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/shobhan126/LinearAlgebraUtils.jl",
    devbranch="main",
)
