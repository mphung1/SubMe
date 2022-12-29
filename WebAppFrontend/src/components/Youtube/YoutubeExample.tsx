//Unused component

import YouTube, { YouTubeProps } from "react-youtube";

function YoutubeExample({ videoId }: { videoId: string }) {
  const onPlayerReady: YouTubeProps["onReady"] = (event) => {
    event.target.pauseVideo();
  };

  const videoOnPlay: YouTubeProps["onPlay"] = (event) => {
    // console.log(event.target.getCurrentTime());
  };
  const videoStateChange: YouTubeProps["onStateChange"] = (event) => {
    // console.log(event.target.getCurrentTime());
  };
  const opts: YouTubeProps["opts"] = {
    height: "390",
    width: "640",
    playerVars: {
      autoplay: 1,
    },
  };
  return (
    <>
      <YouTube
        videoId={videoId}
        opts={opts}
        onReady={onPlayerReady}
        onPlay={videoOnPlay}
        onStateChange={videoStateChange}
      />
    </>
  );
}
