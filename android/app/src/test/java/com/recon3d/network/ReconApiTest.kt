package com.recon3d.network

import kotlinx.coroutines.test.runTest
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import retrofit2.HttpException
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class ReconApiTest {
    private lateinit var server: MockWebServer
    private lateinit var api: ReconApi

    @Before fun setup() {
        server = MockWebServer()
        server.start()
        api = Retrofit.Builder()
            .baseUrl(server.url("/"))
            .client(OkHttpClient())
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ReconApi::class.java)
    }

    @After fun tearDown() = server.shutdown()

    @Test fun `status parses all fields including error`() = runTest {
        server.enqueue(MockResponse().setBody(
            """{"task_id":"a3f9c2","state":"failed","progress":30,"stage_detail":"iteration 100/7000","eta_seconds":null,"error":{"code":"COLMAP_LOW_REGISTRATION","message_zh":"位姿配准率过低"}}"""
        ))
        val status = api.getStatus("a3f9c2")
        assertEquals("a3f9c2", status.taskId)
        assertEquals("failed", status.state)
        assertEquals(30, status.progress)
        assertEquals("iteration 100/7000", status.stageDetail)
        assertNull(status.etaSeconds)
        assertEquals("COLMAP_LOW_REGISTRATION", status.error!!.code)
        assertEquals("位姿配准率过低", status.error!!.messageZh)
        assertEquals("/api/tasks/a3f9c2/status", server.takeRequest().path)
    }

    @Test fun `status without error has null error`() = runTest {
        server.enqueue(MockResponse().setBody(
            """{"task_id":"x","state":"training","progress":55,"stage_detail":"","eta_seconds":180,"error":null}"""
        ))
        val status = api.getStatus("x")
        assertEquals("training", status.state)
        assertEquals(180, status.etaSeconds)
        assertNull(status.error)
    }

    @Test fun `400 error body maps to ApiException with code and message_zh`() = runTest {
        server.enqueue(MockResponse().setResponseCode(400).setBody(
            """{"error":{"code":"SERVER_BUSY","message_zh":"服务器忙"}}"""
        ))
        try {
            api.getStatus("x")
            fail("should throw")
        } catch (e: HttpException) {
            val apiEx = e.toApiException()
            assertEquals("SERVER_BUSY", apiEx.code)
            assertEquals("服务器忙", apiEx.messageZh)
        }
    }

    @Test fun `404 without error body maps to HTTP_404`() = runTest {
        server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"task not found"}"""))
        try {
            api.getStatus("nope")
            fail("should throw")
        } catch (e: HttpException) {
            assertEquals("HTTP_404", e.toApiException().code)
        }
    }

    @Test fun `createTask returns task id`() = runTest {
        server.enqueue(MockResponse().setResponseCode(201).setBody("""{"task_id":"abc123"}"""))
        val part = okhttp3.MultipartBody.Part.createFormData(
            "file", "upload.zip",
            byteArrayOf(1, 2, 3).toRequestBody("application/zip".toMediaTypeOrNull()),
        )
        val resp = api.createTask(part)
        assertEquals("abc123", resp.taskId)
        val recorded = server.takeRequest()
        assertEquals("POST", recorded.method)
        assertEquals("/api/tasks", recorded.path)
        assertTrue(recorded.body.readUtf8().contains("name=\"file\""))
    }
}
