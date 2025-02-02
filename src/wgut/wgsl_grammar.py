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
        Optional(Seq("=", NonTerm("const_expression"))),
    ),
    "global_constant_declaration": Choice(
        Seq(
            "let",
            Choice(NonTerm("identifier"), NonTerm("variable_identifier_declaration")),
            "=",
            NonTerm("const_expression"),
        ),
        Seq(
            Repeat(NonTerm("attribute")),
            "override",
            Choice(NonTerm("identifier"), NonTerm("variable_identifier_declaration")),
            Optional(Seq("=", NonTerm("_expression"))),
        ),
    ),
    "const_expression": Choice(
        Seq(
            NonTerm("type_declaration"),
            "(",
            Optional(
                Seq(
                    Repeat(Seq(NonTerm("const_expression"), ",")),
                    NonTerm("const_expression"),
                    Optional(","),
                )
            ),
            ")",
        ),
        NonTerm("const_literal"),
    ),
}
