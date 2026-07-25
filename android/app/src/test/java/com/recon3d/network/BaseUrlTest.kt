package com.recon3d.network

import org.junit.Assert.assertEquals
import org.junit.Test

class BaseUrlTest {
    @Test fun `bare ip gets scheme and default port`() {
        assertEquals("http://192.168.1.10:8000/", normalizeBaseUrl("192.168.1.10"))
    }

    @Test fun `ip with port keeps port`() {
        assertEquals("http://192.168.1.10:9000/", normalizeBaseUrl("192.168.1.10:9000"))
    }

    @Test fun `full url with trailing slash is unchanged`() {
        assertEquals("http://192.168.1.10:8000/", normalizeBaseUrl("http://192.168.1.10:8000/"))
    }

    @Test fun `whitespace is trimmed`() {
        assertEquals("http://10.0.0.2:8000/", normalizeBaseUrl("  10.0.0.2  "))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `path is rejected`() {
        normalizeBaseUrl("192.168.1.10:8000/api")
    }

    @Test(expected = IllegalArgumentException::class)
    fun `blank is rejected`() {
        normalizeBaseUrl("   ")
    }
}
