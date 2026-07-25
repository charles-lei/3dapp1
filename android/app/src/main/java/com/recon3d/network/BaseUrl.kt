package com.recon3d.network

fun normalizeBaseUrl(input: String): String {
    var s = input.trim().removeSuffix("/")
    require(s.isNotEmpty()) { "服务器地址不能为空" }
    if (!s.startsWith("http://") && !s.startsWith("https://")) s = "http://$s"
    val authority = s.substringAfter("://")
    require(!authority.contains("/")) { "只支持 主机[:端口]，不要填路径" }
    require(authority.isNotBlank()) { "服务器地址不能为空" }
    if (!authority.contains(":")) s = "$s:8000"
    return "$s/"
}
