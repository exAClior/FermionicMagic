using FermionicMagic
using Documenter

DocMeta.setdocmeta!(FermionicMagic, :DocTestSetup, :(using FermionicMagic); recursive=true)

makedocs(;
    modules=[FermionicMagic],
    authors="Yusheng Zhao <yushengzhao2020@outlook.com> and contributors",
    sitename="FermionicMagic.jl",
    format=Documenter.HTML(;
        canonical="https://exAClior.github.io/FermionicMagic.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/exAClior/FermionicMagic.jl",
    devbranch="main",
)
