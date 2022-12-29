import axios from "axios";

export default axios.create({
  baseURL: "https://www.googleapis.com/youtube/v3/",
  params: {
    part: "snippet",
    maxResults: 9,
    key: `${process.env.NEXT_PUBLIC_API_KEY}`,
  },
});
