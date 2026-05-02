package com.example.pokevision_mobile

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class HunterForegroundService : Service() {
    private val wsClient = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var serverUrl: String = ""
    private var connected = false
    private var currentTargets: List<String> = emptyList()
    private var pendingTargets: List<String>? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            HunterContract.ACTION_START -> {
                val url = intent.getStringExtra(HunterContract.EXTRA_SERVER_URL)?.trim().orEmpty()
                if (url.isNotEmpty()) {
                    serverUrl = url
                }
                startForeground(
                    FOREGROUND_ID,
                    buildServiceNotification("Connecting to $serverUrl")
                )
                connect()
            }

            HunterContract.ACTION_SET_TARGETS -> {
                val targets = intent.getStringArrayListExtra(HunterContract.EXTRA_TARGETS)?.toList().orEmpty()
                pendingTargets = targets
                sendTargets(targets)
            }

            HunterContract.ACTION_REQUEST_TARGETS -> {
                webSocket?.send(JSONObject().put("action", "get_targets").toString())
            }

            HunterContract.ACTION_STOP -> {
                disconnect("Stopped from app")
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }

        return START_STICKY
    }

    override fun onDestroy() {
        disconnect("Service destroyed")
        wsClient.dispatcher.executorService.shutdown()
        super.onDestroy()
    }

    private fun connect() {
        if (serverUrl.isBlank()) {
            broadcastState(false, "Missing server URL")
            return
        }

        disconnect("Reconnecting")

        val request = Request.Builder().url(serverUrl).build()
        webSocket = wsClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                connected = true
                updateForeground("Connected to $serverUrl")
                broadcastState(true, "Connected")
                webSocket.send(JSONObject().put("action", "get_targets").toString())
                pendingTargets?.let { sendTargets(it) }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleMessage(text)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                connected = false
                updateForeground("Disconnected")
                broadcastState(false, "Closing: $reason")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                connected = false
                updateForeground("Disconnected")
                broadcastState(false, "Closed: $reason")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                connected = false
                updateForeground("Connection failed")
                broadcastState(false, "Failure: ${t.message ?: "unknown"}")
            }
        })
    }

    private fun disconnect(reason: String) {
        webSocket?.close(1000, reason)
        webSocket = null
        connected = false
    }

    private fun sendTargets(targets: List<String>) {
        val socket = webSocket
        if (!connected || socket == null) {
            broadcastState(false, "Not connected; targets queued")
            return
        }

        val payload = JSONObject()
            .put("action", "set_targets")
            .put("targets", JSONArray(targets))
            .toString()
        socket.send(payload)
    }

    private fun handleMessage(raw: String) {
        val msg = try {
            JSONObject(raw)
        } catch (_: Exception) {
            return
        }

        when (msg.optString("event")) {
            "targets_updated", "targets_current" -> {
                val arr = msg.optJSONArray("targets") ?: JSONArray()
                currentTargets = List(arr.length()) { idx -> arr.optString(idx) }
                broadcastState(connected, "Targets synced")
            }

            "shiny_detected" -> {
                val label = msg.optString("label", "unknown_shiny")
                val score = msg.optDouble("score", 0.0)
                notifyShiny(label, score)
                broadcastShiny(label, score)
            }

            "error" -> {
                val message = msg.optString("message", "Unknown server error")
                broadcastState(connected, "Server error: $message")
            }

            "pong" -> {
                broadcastState(connected, "Connected")
            }
        }
    }

    private fun buildServiceNotification(status: String) =
        NotificationCompat.Builder(this, CHANNEL_STATUS)
            .setContentTitle("Pokevision hunter")
            .setContentText(status)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(appPendingIntent())
            .build()

    private fun updateForeground(status: String) {
        NotificationManagerCompat.from(this).notify(FOREGROUND_ID, buildServiceNotification(status))
    }

    private fun notifyShiny(label: String, score: Double) {
        val notification = NotificationCompat.Builder(this, CHANNEL_ALERTS)
            .setContentTitle("Shiny detected")
            .setContentText("$label (score ${"%.2f".format(score)})")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .setContentIntent(appPendingIntent())
            .build()

        NotificationManagerCompat.from(this).notify(NOTIF_SHINY_BASE + (System.currentTimeMillis() % 10000).toInt(), notification)
    }

    private fun appPendingIntent(): PendingIntent {
        val intent = Intent(this, MainActivity::class.java)
        val flags = PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        return PendingIntent.getActivity(this, 0, intent, flags)
    }

    private fun broadcastState(isConnected: Boolean, status: String) {
        val intent = Intent(HunterContract.ACTION_STATE)
            .setPackage(packageName)
            .putExtra(HunterContract.EXTRA_CONNECTED, isConnected)
            .putExtra(HunterContract.EXTRA_STATUS, status)
            .putStringArrayListExtra(HunterContract.EXTRA_TARGETS, ArrayList(currentTargets))
        sendBroadcast(intent)
    }

    private fun broadcastShiny(label: String, score: Double) {
        val intent = Intent(HunterContract.ACTION_SHINY)
            .setPackage(packageName)
            .putExtra(HunterContract.EXTRA_LABEL, label)
            .putExtra(HunterContract.EXTRA_SCORE, score)
        sendBroadcast(intent)
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager

        val statusChannel = NotificationChannel(
            CHANNEL_STATUS,
            "Hunter status",
            NotificationManager.IMPORTANCE_LOW
        )

        val alertsChannel = NotificationChannel(
            CHANNEL_ALERTS,
            "Shiny alerts",
            NotificationManager.IMPORTANCE_HIGH
        )

        manager.createNotificationChannel(statusChannel)
        manager.createNotificationChannel(alertsChannel)
    }

    companion object {
        private const val CHANNEL_STATUS = "hunter_status"
        private const val CHANNEL_ALERTS = "hunter_alerts"
        private const val FOREGROUND_ID = 1001
        private const val NOTIF_SHINY_BASE = 2000
    }
}
