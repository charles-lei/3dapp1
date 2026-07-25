package com.recon3d.network

import com.google.gson.annotations.SerializedName
import retrofit2.HttpException

data class CreateTaskResponse(@SerializedName("task_id") val taskId: String)

data class ErrorBody(
    val code: String,
    @SerializedName("message_zh") val messageZh: String,
)

data class TaskStatusDto(
    @SerializedName("task_id") val taskId: String,
    val state: String,
    val progress: Int,
    @SerializedName("stage_detail") val stageDetail: String,
    @SerializedName("eta_seconds") val etaSeconds: Int?,
    val error: ErrorBody?,
)

class ApiException(val code: String, val messageZh: String) : Exception("$code: $messageZh")

private data class ErrorEnvelope(val error: ErrorBody?)

fun HttpException.toApiException(): ApiException {
    val parsed = runCatching {
        com.google.gson.Gson().fromJson(response()?.errorBody()?.string(), ErrorEnvelope::class.java).error
    }.getOrNull()
    return ApiException(
        code = parsed?.code ?: "HTTP_${code()}",
        messageZh = parsed?.messageZh ?: "服务器错误（HTTP ${code()}）",
    )
}
