package com.recon3d.network

import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.DELETE
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import retrofit2.http.Streaming

interface ReconApi {
    @Multipart
    @POST("api/tasks")
    suspend fun createTask(@Part file: MultipartBody.Part): CreateTaskResponse

    @GET("api/tasks/{id}/status")
    suspend fun getStatus(@Path("id") id: String): TaskStatusDto

    @Streaming
    @GET("api/tasks/{id}/model")
    suspend fun downloadModel(@Path("id") id: String): ResponseBody

    @DELETE("api/tasks/{id}")
    suspend fun deleteTask(@Path("id") id: String): Response<Unit>
}
