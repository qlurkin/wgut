from grammar import Optional, Seq, Choice, NonTerm, Repeat

WgslGrammar = {
    "start": Seq(Repeat(NonTerm("enable_directive")), Repeat(NonTerm("_declaration"))),
    "_declaration": Choice(
        ";",
        Seq(NonTerm("global_variable_declaration"), ";"),
        Seq(NonTerm("global_constant_declaration"), ";"),
        Seq(NonTerm("type_alias_declaration"), ";"),
        NonTerm("struct_declaration"),
        NonTerm("function_declaration"),
    ),
    "global_variable_declaration": Seq(
        Repeat(NonTerm("attribute")),
        NonTerm("variable_declaration"),
        Optional(Seq("=", NonTerm("const_expr"))),
    ),
}
